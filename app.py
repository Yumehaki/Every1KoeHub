# app.py
import os
import secrets
import hashlib
from typing import Optional, List, Tuple
import uuid as uuidlib
import mimetypes
import urllib.parse
import zipfile
import json
from io import BytesIO
from datetime import datetime, timezone, timedelta

from fastapi import (
    FastAPI, Request, Depends, Form, UploadFile, File, HTTPException
)
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import PlainTextResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import (
    create_engine, Column, String, DateTime, Text, Boolean,
    ForeignKey, Table, BigInteger, Integer, UniqueConstraint, func
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session as OrmSession
from passlib.hash import bcrypt
import shutil
# -----------------------------------
# 設定
# -----------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")
MODEL_STORAGE_DIR = os.environ.get("MODEL_STORAGE_DIR", "./storage/models")
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
SAFE_STORAGE_DIR = os.environ.get("SAFE_STORAGE_DIR", "./storage/quarantine")  # 管理削除時の退避先（DL不可領域）
BACKUP_DIR = os.environ.get("BACKUP_DIR", "./storage/backups")                 # DBバックアップ保存先
BANNED_WORDS = [w.strip() for w in os.environ.get("BANNED_WORDS", "非公式,無許可,Anneli").split(",") if w.strip()]

os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
os.makedirs(SAFE_STORAGE_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
# -----------------------------------
# DB接続
# -----------------------------------
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

def new_uuid() -> str:
    return str(uuidlib.uuid4())

def _ensure_sqlite_columns():
    """
    既存SQLite DB に不足カラムがある場合に、起動時に軽量ALTERで追加する。
    - users: email_verified (bool), email_verification_token (str)
    - models: is_locked (bool), is_certified (bool)
    既に存在する場合はスキップ。
    ※ SQLite前提の簡易実装。その他DBはマイグレーションツールを推奨。
    """
    if not DATABASE_URL.startswith("sqlite"):
        return
    path = DATABASE_URL.replace("sqlite:///", "", 1).replace("sqlite:////", "/", 1)
    if not os.path.exists(path):
        return
    import sqlite3
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        # users テーブル
        cur.execute("PRAGMA table_info(users)")
        user_cols = {row[1] for row in cur.fetchall()}  # name in 2nd col
        if "email_verified" not in user_cols:
            cur.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 0")
        if "email_verification_token" not in user_cols:
            cur.execute("ALTER TABLE users ADD COLUMN email_verification_token TEXT DEFAULT ''")
        # models テーブル
        cur.execute("PRAGMA table_info(models)")
        model_cols = {row[1] for row in cur.fetchall()}
        if "is_locked" not in model_cols:
            cur.execute("ALTER TABLE models ADD COLUMN is_locked INTEGER DEFAULT 0")
        if "is_certified" not in model_cols:
            cur.execute("ALTER TABLE models ADD COLUMN is_certified INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()

# 先に不足カラムを補う（SQLite）
_ensure_sqlite_columns()

# -----------------------------------
# 中間テーブル（UUID参照）
# -----------------------------------
unit_models = Table(
    "unit_models",
    Base.metadata,
    Column("unit_uuid", String(36), ForeignKey("units.uuid", ondelete="CASCADE"), primary_key=True),
    Column("model_uuid", String(36), ForeignKey("models.uuid", ondelete="CASCADE"), primary_key=True),
    UniqueConstraint("unit_uuid", "model_uuid"),
)

# -----------------------------------
# モデル定義（主キーは全てUUID）
# -----------------------------------
class User(Base):
    __tablename__ = "users"
    uuid = Column(String(36), primary_key=True, default=new_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    bio = Column(Text, default="")
    icon_path = Column(String(255), default="")
    is_admin = Column(Boolean, default=False)
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(64), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    models = relationship("Model", back_populates="author", primaryjoin="User.uuid==Model.user_uuid", cascade="all,delete")
    likes = relationship("Like", back_populates="user", cascade="all,delete-orphan")

class Model(Base):
    __tablename__ = "models"
    uuid = Column(String(36), primary_key=True, default=new_uuid)
    user_uuid = Column(String(36), ForeignKey("users.uuid", ondelete="SET NULL"), index=True, nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    license_name = Column(String(64), default="Custom")
    license_text = Column(Text, default="")

    # ファイル本体
    file_path = Column(String(1024), nullable=False)
    size_bytes = Column(BigInteger, default=0)
    sha256 = Column(String(64), index=True)

    # 追加メディア（自動抽出/手動設定どちらもOK）
    icon_path = Column(String(1024), default="")
    sample_icon_path = Column(String(1024), default="")
    sample_audio_path = Column(String(1024), default="")

    downloads = Column(Integer, default=0)
    likes_count = Column(Integer, default=0)
    is_locked = Column(Boolean, default=False)     # 施錠（DL不可・表示は可）
    is_certified = Column(Boolean, default=False)  # 運営による認証マーク
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    author = relationship("User", back_populates="models", foreign_keys=[user_uuid])
    units = relationship("Unit", secondary=unit_models, back_populates="models")
    likes = relationship("Like", back_populates="model", cascade="all,delete-orphan")

class ModelSampleMedia(Base):
    """
    モデルに紐づく「複数」サンプルメディア（画像/音声）
    - kind: 'image' or 'audio'
    - path: 保存先の実ファイルパス
    - sort_order: 表示順
    """
    __tablename__ = "model_sample_media"
    uuid = Column(String(36), primary_key=True, default=new_uuid)
    model_uuid = Column(String(36), ForeignKey("models.uuid", ondelete="CASCADE"), index=True, nullable=False)
    kind = Column(String(16), nullable=False)  # "image" | "audio"
    path = Column(String(1024), nullable=False)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    model = relationship("Model", primaryjoin="ModelSampleMedia.model_uuid==Model.uuid")

class Unit(Base):
    __tablename__ = "units"
    uuid = Column(String(36), primary_key=True, default=new_uuid)
    owner_user_uuid = Column(String(36), ForeignKey("users.uuid", ondelete="SET NULL"), index=True, nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    icon_path = Column(String(1024), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    models = relationship("Model", secondary=unit_models, back_populates="units")

class Like(Base):
    __tablename__ = "likes"
    uuid = Column(String(36), primary_key=True, default=new_uuid)
    user_uuid = Column(String(36), ForeignKey("users.uuid", ondelete="CASCADE"), index=True)
    model_uuid = Column(String(36), ForeignKey("models.uuid", ondelete="CASCADE"), index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (UniqueConstraint("user_uuid", "model_uuid", name="uix_user_model_like"),)

    user = relationship("User", back_populates="likes")
    model = relationship("Model", back_populates="likes")

class Announcement(Base):
    __tablename__ = "announcements"
    uuid = Column(String(36), primary_key=True, default=new_uuid)
    admin_user_uuid = Column(String(36), ForeignKey("users.uuid", ondelete="SET NULL"))
    title = Column(String(255), nullable=False)
    content = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# -----------------------------------
# 依存
# -----------------------------------

# 起動時にテーブルが存在することを再確認（checkfirst=True で安全）
def get_db() -> OrmSession:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------
# 認証ヘルパ
# -----------------------------------
def get_current_user(request: Request, db: OrmSession) -> Optional[User]:
    uid = request.session.get("user_uuid")
    if not uid:
        return None
    return db.query(User).filter(User.uuid == uid).first()

def login_required(user: Optional[User]):
    if not user:
        raise HTTPException(status_code=401, detail="Login required")

# -----------------------------------
# ユーティリティ
# -----------------------------------
def hash_password(pw: str) -> str:
    return bcrypt.hash(pw)

def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.verify(pw, hashed)

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_stream_and_hash(dst_path: str, up: UploadFile) -> Tuple[int, str]:
    """大容量に耐えるチャンク保存とSHA-256算出"""
    ensure_dir(dst_path)
    sha = hashlib.sha256()
    size = 0
    with open(dst_path, "wb") as f:
        while True:
            chunk = up.file.read(1024 * 1024 * 16)
            if not chunk:
                break
            f.write(chunk)
            sha.update(chunk)
            size += len(chunk)
    return size, sha.hexdigest()

def save_small_file(dst_path: str, up: UploadFile) -> str:
    """画像や音声などの比較的小さなファイル保存"""
    init_data = up.file.read(1)
    if not init_data:
        return dst_path
    ensure_dir(dst_path)
    with open(dst_path, "wb") as f:
        f.write(init_data)
        while True:
            chunk = up.file.read(1024 * 256) # 256KBごと
            if not chunk:
                break
            f.write(chunk)
    return dst_path

def guess_mime(path: str, default: str = "application/octet-stream") -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or default

def ascii_fallback(name: str) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 and ch not in "\\\";" else "_" for ch in name)

def build_content_disposition(filename: str, inline: bool = False) -> str:
    """
    RFC 5987 対応の Content-Disposition を返す。
    - ASCII フォールバック filename
    - UTF-8 % エンコードの filename* も同時付与
    """
    disp = "inline" if inline else "attachment"
    fallback = ascii_fallback(filename or "download.bin")
    quoted = urllib.parse.quote(filename or "download.bin")
    return f"{disp}; filename={fallback}; filename*=UTF-8''{quoted}"

def model_dir(model_uuid: str) -> str:
    d = os.path.join(MODEL_STORAGE_DIR, model_uuid)
    os.makedirs(d, exist_ok=True)
    return d

# LIKE検索のエスケープ
def escape_like(term: str) -> str:
    # \, %, _ をエスケープ
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

# -----------------------------------
# .aivmx からの抽出
# -----------------------------------
def extract_from_aivmx(aivmx_path: str, out_dir: str) -> dict:
    """
    .aivmx を zip として開き、以下を探索して保存する。
      - icon.(png|jpg|jpeg)
      - sample_icon.(png|jpg|jpeg)
      - sample_audio.(wav|wav|m4a|ogg)
      - manifest.json の base64 埋め込み（icon_base64, sample_icon_base64, sample_audio_base64 等）
    戻り値: {"icon_path": str|"" , "sample_icon_path": str|"" , "sample_audio_path": str|""}
    """
    result = {"icon_path": "", "sample_icon_path": "", "sample_audio_path": ""}

    if not os.path.isfile(aivmx_path):
        return result

    try:
        with zipfile.ZipFile(aivmx_path, "r") as z:
            names = z.namelist()
            # ファイル名探索
            def extract_first(candidates, outname):
                for n in names:
                    low = n.lower()
                    for cand in candidates:
                        if low.endswith(cand):
                            data = z.read(n)
                            path = os.path.join(out_dir, outname)
                            ensure_dir(path)
                            with open(path, "wb") as f:
                                f.write(data)
                            return path
                return ""

            # 直接ファイル
            icon = extract_first(["icon.png", "icon.jpg", "icon.jpeg"], "ex_icon.png")
            sicon = extract_first(["sample_icon.png", "sample_icon.jpg", "sample_icon.jpeg"], "ex_sample_icon.png")
            saudio = extract_first(["sample_audio.wav", "sample_audio.wav", "sample_audio.m4a", "sample_audio.ogg"], "ex_sample_audio")

            # manifest.json も見る
            if "manifest.json" in [n.lower() for n in names]:
                manifest_name = [n for n in names if n.lower() == "manifest.json"][0]
                try:
                    manifest = json.loads(z.read(manifest_name).decode("utf-8", errors="ignore"))
                    import base64
                    if not icon:
                        b64 = manifest.get("icon_base64") or manifest.get("icon")
                        if isinstance(b64, str):
                            raw = base64.b64decode(b64)
                            path = os.path.join(out_dir, "ex_icon_from_manifest.png")
                            with open(path, "wb") as f:
                                f.write(raw)
                            icon = path
                    if not sicon:
                        b64 = manifest.get("sample_icon_base64") or manifest.get("sample_icon")
                        if isinstance(b64, str):
                            raw = base64.b64decode(b64)
                            path = os.path.join(out_dir, "ex_sample_icon_from_manifest.png")
                            with open(path, "wb") as f:
                                f.write(raw)
                            sicon = path
                    if not saudio:
                        b64 = manifest.get("sample_audio_base64") or manifest.get("sample_audio")
                        if isinstance(b64, str):
                            raw = base64.b64decode(b64)
                            path = os.path.join(out_dir, "ex_sample_audio_from_manifest")
                            with open(path, "wb") as f:
                                f.write(raw)
                            saudio = path
                except Exception:
                    pass

            result["icon_path"] = icon or ""
            result["sample_icon_path"] = sicon or ""
            result["sample_audio_path"] = saudio or ""
    except Exception:
        # 壊れているなど
        pass

    return result

# -----------------------------------
# アプリ
# -----------------------------------
app = FastAPI(title="Every1Koe Hub")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, session_cookie="sessionid", max_age=3600*24*7)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------
# セッション依存関数
# -------------------------
@app.on_event("startup")
def _ensure_tables_on_startup():
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        # 念のため新設テーブルも個別に確認
        try:
            ModelSampleMedia.__table__.create(bind=engine, checkfirst=True)
        except Exception:
            pass
    except Exception:
        pass

# -------------------------
# ユーザアイコンの配信（プレビュー用）
# -------------------------
@app.get("/media/users/{user_uuid}/icon")
def media_user_icon(user_uuid: str, db: OrmSession = Depends(get_db)):
    u = db.query(User).filter(User.uuid == user_uuid).first()
    if not u or not u.icon_path:
        raise HTTPException(404, "No user icon")
    headers = {"Content-Disposition": build_content_disposition(os.path.basename(u.icon_path), inline=True)}
    return StreamingResponse(open(u.icon_path, "rb"),
                             media_type=guess_mime(u.icon_path, "image/png"),
                             headers=headers)

# -----------------------------------
# トップページ
# -----------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: OrmSession = Depends(get_db)):
    popular = db.query(Model).filter(Model.is_public == True).order_by(Model.likes_count.desc(), Model.created_at.desc()).limit(12).all()
    most_dl = db.query(Model).filter(Model.is_public == True).order_by(Model.downloads.desc(), Model.created_at.desc()).limit(12).all()
    newest = db.query(Model).filter(Model.is_public == True).order_by(Model.created_at.desc()).limit(12).all()
    announcements = db.query(Announcement).order_by(Announcement.created_at.desc()).limit(10).all()
    user = get_current_user(request, db)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "popular": popular,
        "most_dl": most_dl,
        "newest": newest,
        "announcements": announcements
    })

# -----------------------------------
# 認証
# -----------------------------------
@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("auth_register.html", {"request": request})

@app.post("/register")
def register(
    request: Request,
    email: EmailStr = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db: OrmSession = Depends(get_db)
):
    if db.query(User).filter((User.email == email) | (User.username == username)).first():
        raise HTTPException(400, "Email or Username already used")
    u = User(email=str(email), username=username, password_hash=hash_password(password))
    db.add(u); db.commit()
    request.session["user_uuid"] = u.uuid
    return RedirectResponse(url="/", status_code=302)
# -------------------------
# メール認証（簡易）: トークン発行と確認
# -------------------------
@app.post("/verify/start")
def start_email_verification(request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    if user.email_verified:
        return RedirectResponse(url=f"/users/{user.uuid}", status_code=302)
    token = secrets.token_hex(16)
    user.email_verification_token = token
    db.add(user); db.commit()
    # 実運用ではメール送信する。ここでは画面でトークンを返す簡易実装。
    return PlainTextResponse(f"Verification token (for demo): {token}\nConfirm at: /verify/confirm?token={token}")

@app.get("/verify/confirm")
def confirm_email_verification(token: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    if not token or token != user.email_verification_token:
        raise HTTPException(400, "Invalid token")
    user.email_verified = True
    user.email_verification_token = ""
    db.add(user); db.commit()
    return RedirectResponse(url=f"/users/{user.uuid}", status_code=302)

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("auth_login.html", {"request": request})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), db: OrmSession = Depends(get_db)):
    u = db.query(User).filter(User.username == username).first()
    if not u or not verify_password(password, u.password_hash):
        raise HTTPException(400, "Invalid credentials")
    request.session["user_uuid"] = u.uuid
    return RedirectResponse(url="/", status_code=302)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)

# -----------------------------------
# ユーザー関連
# -----------------------------------
@app.get("/users/{user_uuid}", response_class=HTMLResponse)
def user_page(user_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    u = db.query(User).filter(User.uuid == user_uuid).first()
    if not u:
        raise HTTPException(404, "User not found")
    models = db.query(Model).filter(Model.user_uuid == u.uuid, Model.is_public == True).order_by(Model.created_at.desc()).all()
    units = db.query(Unit).filter(Unit.owner_user_uuid == u.uuid).order_by(Unit.created_at.desc()).all()
    cur_user = get_current_user(request, db)
    return templates.TemplateResponse("user_profile.html", {
        "request": request, "profile": u, "user": cur_user, "models": models, "units": units
    })

@app.get("/users/me/edit", response_class=HTMLResponse)
def user_edit_form(request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    return templates.TemplateResponse("user_edit.html", {"request": request, "user": user})

@app.post("/users/me/edit")
def user_edit(
    request: Request,
    # 空欄は据え置きにするため Optional に変更
    bio: Optional[str] = Form(None),
    icon: Optional[UploadFile] = File(None),
    db: OrmSession = Depends(get_db)
):
    user = get_current_user(request, db); login_required(user)
    # 空欄なら変更しない
    user.bio = bio
    if icon:
        ext = os.path.splitext(icon.filename or "")[1] or ".png"
        dst = os.path.join(MODEL_STORAGE_DIR, "users", user.uuid, f"icon{ext}")
        save_small_file(dst, icon)
        user.icon_path = dst
    db.add(user); db.commit()
    return RedirectResponse(url=f"/users/{user.uuid}", status_code=302)

# -----------------------------------
# モデル：アップロード（自動抽出＋手動上書き可）
# -----------------------------------
LICENSE_TEMPLATES = {
    "カスタムライセンス":"Custom",  # Custom時のみ自由入力を使用
    "ACML 1.0":"",
    "ACML-NC 1.0":"",
    "EKCML 1.0":"",
    "EKCML-NC 1.0":"",
    "EKCML-ND 1.0":"",
    "EKCML-MD 1.0":"",
    "EKCML-ND-NC 1.0":"",
    "EKCML-MD-NC 1.0":"",
    "CC0":"",
    "Yumehaki Public License 1.0":"",
}

@app.get("/models/upload", response_class=HTMLResponse)
def upload_model_form(request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    return templates.TemplateResponse("model_upload.html", {"request": request, "user": user, "license_templates": LICENSE_TEMPLATES})

@app.post("/models/upload")
def upload_model(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    license_name: str = Form("Custom"),
    license_text: str = Form(""),
    file: UploadFile = File(...),
    # 手動メディア（抽出失敗 or 上書き用、任意）
    icon: Optional[UploadFile] = File(None),
    sample_icon: Optional[UploadFile] = File(None),   # 従来の単一（後方互換）
    sample_audio: Optional[UploadFile] = File(None),  # 従来の単一（後方互換）
    db: OrmSession = Depends(get_db)
):
    user = get_current_user(request, db); login_required(user)
    # NGワード（複数）を含むモデル名は拒否（大文字小文字を区別しない簡易判定）
    low = (name or "").lower()
    for w in BANNED_WORDS:
        if w and w.lower() in low:
            raise HTTPException(400, f"投稿拒否: モデル名に禁止語 '{w}' が含まれています")

    model_uuid = new_uuid()
    # 本体
    original = os.path.basename(file.filename or "model.aivmx")
    dst_main = os.path.join(model_dir(model_uuid), f"{model_uuid}_{original}")
    size, sha256 = save_stream_and_hash(dst_main, file)

    # ライセンス本文
    tpl_text = next((t[1] for t in LICENSE_TEMPLATES if t[0] == license_name), "")
    final_license_text = license_text if license_name == "Custom" or license_name == "カスタムライセンス" else tpl_text

    m = Model(
        uuid=model_uuid, user_uuid=user.uuid, name=name, description=description,
        license_name=license_name, license_text=final_license_text,
        file_path=dst_main, size_bytes=size, sha256=sha256, is_public=True
    )

    # .aivmx抽出
    ex = extract_from_aivmx(dst_main, model_dir(model_uuid))

    # 自動抽出（存在すればセット）
    if ex.get("icon_path"): m.icon_path = ex["icon_path"]
    if ex.get("sample_icon_path"): m.sample_icon_path = ex["sample_icon_path"]
    if ex.get("sample_audio_path"): m.sample_audio_path = ex["sample_audio_path"]

    # 手動アップがあれば上書き（優先）
    if icon:
        icon_name = f"{model_uuid}_icon{os.path.splitext(icon.filename or '')[1] or '.png'}"
        m.icon_path = save_small_file(os.path.join(model_dir(model_uuid), icon_name), icon)
    if sample_icon:
        s_icon_name = f"{model_uuid}_sample_icon{os.path.splitext(sample_icon.filename or '')[1] or '.png'}"
        m.sample_icon_path = save_small_file(os.path.join(model_dir(model_uuid), s_icon_name), sample_icon)
    if sample_audio:
        s_audio_ext = os.path.splitext(sample_audio.filename or "")[1] or ".wav"
        s_audio_name = f"{model_uuid}_sample_audio{s_audio_ext}"
        m.sample_audio_path = save_small_file(os.path.join(model_dir(model_uuid), s_audio_name), sample_audio)

    db.add(m); db.commit()
    return RedirectResponse(url=f"/models/{m.uuid}", status_code=302)

# -----------------------------------
# モデル：詳細
# -----------------------------------
@app.get("/models/{model_uuid}", response_class=HTMLResponse)
def model_detail(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or (not m.is_public and (not get_current_user(request, db) or get_current_user(request, db).uuid != m.user_uuid)):
        raise HTTPException(404, "Model not found")
    user = get_current_user(request, db)
    # 複数サンプルを取得（kind順・sort_order順）
    sample_images = db.query(ModelSampleMedia)\
        .filter(ModelSampleMedia.model_uuid==m.uuid, ModelSampleMedia.kind=="image")\
        .order_by(ModelSampleMedia.sort_order.asc(), ModelSampleMedia.created_at.asc()).all()
    sample_audios = db.query(ModelSampleMedia)\
        .filter(ModelSampleMedia.model_uuid==m.uuid, ModelSampleMedia.kind=="audio")\
        .order_by(ModelSampleMedia.sort_order.asc(), ModelSampleMedia.created_at.asc()).all()
    return templates.TemplateResponse("model_detail.html", {
        "request": request, "user": user, "model": m, "units": m.units,
        "sample_images": sample_images, "sample_audios": sample_audios
    })

# -----------------------------------
# モデル：編集・再抽出・削除
# -----------------------------------
@app.get("/models/{model_uuid}/edit", response_class=HTMLResponse)
def model_edit_form(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or m.user_uuid != user.uuid:
        raise HTTPException(403, "Forbidden")
        # 既存の複数サンプル（音声）を編集画面で表示・削除できるように取得
    edit_sample_audios = db.query(ModelSampleMedia)\
        .filter(
            ModelSampleMedia.model_uuid == m.uuid,
            ModelSampleMedia.kind == "audio"
        )\
        .order_by(ModelSampleMedia.sort_order.asc(), ModelSampleMedia.created_at.asc())\
        .all()
    ctx = {
        "request": request,
        "user": user,
        "model": m,
        "license_templates": LICENSE_TEMPLATES,
        "model_icon_available": bool(m.icon_path),
        "model_icon_url": f"/media/models/{m.uuid}/icon" if m.icon_path else "",
        "sample_icon_available": bool(m.sample_icon_path),
        "sample_icon_url": f"/media/models/{m.uuid}/sample-icon" if m.sample_icon_path else "",
        "sample_audio_available": bool(m.sample_audio_path),
        "sample_audio_url": f"/media/models/{m.uuid}/sample-audio" if m.sample_audio_path else "",
        "edit_sample_audios": edit_sample_audios,
    }
    return templates.TemplateResponse("model_edit.html", ctx)

@app.post("/models/{model_uuid}/samples/delete")
def delete_model_sample_audios(
    model_uuid: str,
    request: Request,
    delete_audio_ids: List[str] = Form([]),
    db: OrmSession = Depends(get_db)
):
    """
    所有者が指定した複数のサンプル音声（ModelSampleMedia, kind='audio'）を削除。
    - DB行削除
    - 実ファイル削除（best-effort）
    """
    user = get_current_user(request, db); login_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or m.user_uuid != user.uuid:
        raise HTTPException(403, "Not allowed")
    if not delete_audio_ids:
        return RedirectResponse(url=f"/models/{m.uuid}/edit", status_code=302)
    items = db.query(ModelSampleMedia).filter(
        ModelSampleMedia.model_uuid == m.uuid,
        ModelSampleMedia.kind == "audio",
        ModelSampleMedia.uuid.in_(delete_audio_ids)
    ).all()
    for it in items:
        try:
            if it.path and os.path.exists(it.path):
                os.remove(it.path)
        except Exception:
            pass
        db.delete(it)
    db.commit()
    return RedirectResponse(url=f"/models/{m.uuid}/edit", status_code=302)

@app.post("/models/{model_uuid}/edit")
def model_edit(
    model_uuid: str, request: Request,
    # 空欄は据え置きにするため Optional に変更
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    license_name: Optional[str] = Form(None),
    license_text: Optional[str] = Form(None),
    # 公開/非公開の確定値を受け取る（hiddenとcheckboxの併用を想定）
    # 例: hidden name=is_public value="false", checkbox name=is_public value="true"
    is_public: Optional[List[str]] = Form(None),
    # 追加：複数のサンプルメディアを受け付け
    sample_images: Optional[List[UploadFile]] = File(None),
    sample_audios: Optional[List[UploadFile]] = File(None),
    icon: Optional[UploadFile] = File(None),
    sample_icon: Optional[UploadFile] = File(None),
    sample_audio: Optional[UploadFile] = File(None),
    db: OrmSession = Depends(get_db)
):
    user = get_current_user(request, db); login_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or m.user_uuid != user.uuid:
        raise HTTPException(403, "Forbidden")
    # 空欄は変更しない（据え置き）
    # 先に名前のNGワードチェック（変更が指定されている場合のみ）
    if name is not None and name.strip() != "":
        low = name.lower()
        for w in BANNED_WORDS:
            if w and w.lower() in low:
                raise HTTPException(400, f"変更拒否: モデル名に禁止語 '{w}' が含まれています")
    if name is not None and name.strip() != "":
        m.name = name
    if description is not None and description.strip() != "":
        m.description = description
    if license_name is not None and license_name.strip() != "":
        m.license_name = license_name
        # ライセンス本文は Custom 指定時などにのみ上書きしたいケースがあるため、
        # license_text が空欄でなければ上書き、空欄なら据え置き
        if license_text is not None and license_text.strip() != "":
            m.license_text = license_text
        else:
            # license_name を変えたが本文が空のときは既存を保持（テンプレ自動上書きはしない）
            pass
    elif license_text is not None and license_text.strip() != "":
        # ライセンス名を変えずに本文だけ入っている場合は本文のみ更新
        m.license_text = license_text
    # 公開/非公開：hidden(false) と checkbox(true) の併用を許可。
    # 送信された配列の最後の値を採用（ブラウザは後勝ちになることが多い）
    if is_public is not None and len(is_public) > 0:
        val = is_public[-1].lower()
        if val in ("true","1","on","yes"):
            m.is_public = True
        elif val in ("false","0","off","no"):
            m.is_public = False
    m.license_name = license_name
    tpl_text = next((t[1] for t in LICENSE_TEMPLATES if t[0] == license_name), "")
    m.license_text = license_text if license_name == "Custom" or license_name == "カスタムライセンス" else tpl_text
    # 手動での再設定（任意）
    if icon:
        ext = os.path.splitext(icon.filename or "")[1] or ".png"
        dst = os.path.join(model_dir(m.uuid), f"{m.uuid}_icon{ext}")
        m.icon_path = save_small_file(dst, icon)
    m.updated_at = datetime.now(timezone.utc)
    # 施錠/認証は運営UIからのみ変更（ここでは触らない）
    # m.is_locked / m.is_certified は変更しない
    # 既存の単一フィールド（後方互換用）での上書きは従来通り
    # if len(icon.filename) > 0:
    #     ic_name = f"{m.uuid}_icon{os.path.splitext(icon.filename or '')[1] or '.png'}"
    #     m.icon_path = save_small_file(os.path.join(model_dir(m.uuid), ic_name), icon)
    # if len(sample_icon.filename) > 0:
    #     sic_name = f"{m.uuid}_sample_icon{os.path.splitext(sample_icon.filename or '')[1] or '.png'}"
    #     m.sample_icon_path = save_small_file(os.path.join(model_dir(m.uuid), sic_name), sample_icon)
    # if len(sample_audio.filename) > 0:
    #     sa_ext = os.path.splitext(sample_audio.filename or "")[1] or ".wav"
    #     sa_name = f"{m.uuid}_sample_audio{sa_ext}"
    #     m.sample_audio_path = save_small_file(os.path.join(model_dir(m.uuid), sa_name), sample_audio)
    # 新規：複数サンプルの登録
    def _append_media(model_uuid: str, files: Optional[UploadFile], kind: str):
        if not files:
            return
        cur_max = db.query(func.max(ModelSampleMedia.sort_order))\
            .filter(ModelSampleMedia.model_uuid==model_uuid, ModelSampleMedia.kind==kind).scalar() or 0
        order = int(cur_max)
        for f in files:
            try:
                if not f or not (f.filename or "").strip():
                    continue
                order += 1
                ext = os.path.splitext(f.filename or "")[1] or (".png" if kind=="image" or kind=="icon" else ".wav")
                if kind != "icon":
                    dst = os.path.join(model_dir(model_uuid), f"{model_uuid}_{kind}_{order}{ext}")
                else:
                    dst = os.path.join(model_dir(model_uuid), f"{model_uuid}_icon{ext}")
                save_small_file(dst, f)
                db.add(ModelSampleMedia(model_uuid=model_uuid, kind=kind, path=dst, sort_order=order))
            except Exception:
                # 個別の失敗はスキップ（他のファイル継続）
                order -= 0  # 明示的に順番維持（失敗分は欠番許容）

    _append_media(model_uuid, sample_audios, "audio")
    # _append_media(model_uuid, sample_audio, "audio")

    # 新規：複数サンプルの追加（任意・追加分だけを登録）
    def _append_media(files: Optional[List[UploadFile]], kind: str):
        if not files: return
        # 既存の最大 sort_order を取得してインクリメント
        cur_max = db.query(func.max(ModelSampleMedia.sort_order)).filter(ModelSampleMedia.model_uuid==m.uuid, ModelSampleMedia.kind==kind).scalar() or 0
        order = int(cur_max)
        for f in files:
            try:
                if not f or not (f.filename or "").strip():
                    continue
                order += 1
                ext = os.path.splitext(f.filename or "")[1] or (".png" if kind=="image" else ".wav")
                dst = os.path.join(model_dir(m.uuid), f"{m.uuid}_{kind}_{order}{ext}")
                save_small_file(dst, f)
                db.add(ModelSampleMedia(model_uuid=m.uuid, kind=kind, path=dst, sort_order=order))
            except Exception:
                # 個別失敗はスキップ（他のファイルは継続）
                order -= 0  # 明示的に順番維持（失敗分は欠番許容）

    # _append_media(sample_images, "image")
    # _append_media(sample_audios, "audio")
    db.add(m); db.commit()
    return RedirectResponse(url=f"/models/{m.uuid}", status_code=302)

@app.post("/models/{model_uuid}/reextract")
def model_reextract(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or m.user_uuid != user.uuid:
        raise HTTPException(403, "Forbidden")
    ex = extract_from_aivmx(m.file_path, model_dir(m.uuid))
    if ex.get("icon_path"): m.icon_path = ex["icon_path"]
    if ex.get("sample_icon_path"): m.sample_icon_path = ex["sample_icon_path"]
    if ex.get("sample_audio_path"): m.sample_audio_path = ex["sample_audio_path"]
    m.updated_at = datetime.now(timezone.utc)
    db.add(m); db.commit()
    return RedirectResponse(url=f"/models/{m.uuid}", status_code=302)

@app.post("/models/{model_uuid}/delete")
def model_delete(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or m.user_uuid != user.uuid:
        raise HTTPException(403, "Forbidden")
    # ファイル削除
    try:
        if os.path.isdir(model_dir(m.uuid)):
            for root, dirs, files in os.walk(model_dir(m.uuid), topdown=False):
                for f in files:
                    try: os.remove(os.path.join(root, f))
                    except: pass
                for d in dirs:
                    try: os.rmdir(os.path.join(root, d))
                    except: pass
            os.rmdir(model_dir(m.uuid))
    except Exception:
        pass
    db.delete(m); db.commit()
    return RedirectResponse(url=f"/users/{user.uuid}", status_code=302)
# =========================
# モデル：メディア配信
# =========================
def _can_view_model(m: "Model", request: Request, db: OrmSession) -> bool:
    """
    非公開モデルのメディア可視性
    """
    if not m:
        return False
    if m.is_public:
        return True
    cur = get_current_user(request, db)
    if not cur:
        return False
    if cur.is_admin:
        return True
    return cur.uuid == m.user_uuid
# -----------------------------------
# メディア配信（画像/音声）— inline 表示
# -----------------------------------
@app.get("/media/models/{model_uuid}/icon")
def media_model_icon(model_uuid: str, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or not m.icon_path:
        raise HTTPException(404, "No icon")
    headers = {"Content-Disposition": build_content_disposition(os.path.basename(m.icon_path), inline=True)}
    return StreamingResponse(open(m.icon_path, "rb"), media_type=guess_mime(m.icon_path, "image/png"), headers=headers)

@app.get("/media/models/{model_uuid}/sample-icon")
def media_model_sample_icon(model_uuid: str, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or not m.sample_icon_path:
        raise HTTPException(404, "No sample icon")
    headers = {"Content-Disposition": build_content_disposition(os.path.basename(m.sample_icon_path), inline=True)}
    return StreamingResponse(open(m.sample_icon_path, "rb"), media_type=guess_mime(m.sample_icon_path, "image/png"), headers=headers)

@app.get("/media/models/{model_uuid}/sample-audio")
def media_model_sample_audio(model_uuid: str, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or not m.sample_audio_path:
        raise HTTPException(404, "No sample audio")
    headers = {"Content-Disposition": build_content_disposition(os.path.basename(m.sample_audio_path), inline=True)}
    return StreamingResponse(open(m.sample_audio_path, "rb"), media_type=guess_mime(m.sample_audio_path, "audio/mpeg"), headers=headers)

# 追加：複数サンプルの配信（個別UUID指定）
@app.get("/media/models/{model_uuid}/samples/{media_uuid}")
def media_model_sample_item(model_uuid: str, media_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or not _can_view_model(m, request, db):
        raise HTTPException(404, "Not found")
    item = db.query(ModelSampleMedia).filter(
        ModelSampleMedia.uuid == media_uuid,
        ModelSampleMedia.model_uuid == m.uuid
    ).first()
    if not item or not os.path.exists(item.path):
        raise HTTPException(404, "Not found")
    headers = {"Content-Disposition": build_content_disposition(os.path.basename(item.path), inline=True)}
    return StreamingResponse(
        open(item.path, "rb"),
        media_type=guess_mime(item.path, "application/octet-stream"),
        headers=headers
    )

# -----------------------------------
# モデル：ダウンロード（人間向け）
# -----------------------------------
@app.get("/download/{model_uuid}")
def human_download(model_uuid: str, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid, Model.is_public == True).first()
    if not m:
        raise HTTPException(404, "Model not found")
    # 施錠中はDL不可
    if m.is_locked:
        # 表示は可能だがDLは禁止
        raise HTTPException(status_code=403, detail="Locked model: download disabled")
    m.downloads += 1
    m.updated_at = datetime.now(timezone.utc)
    db.add(m); db.commit()

    filename = os.path.basename(m.file_path)
    headers = {"Content-Disposition": build_content_disposition(filename, inline=False)}
    return StreamingResponse(open(m.file_path, "rb"), media_type="application/octet-stream", headers=headers)
# -------------------------
# 運営：モデル施錠/解除・認証付与/剥奪・削除
# -------------------------
def admin_required(user: Optional[User]):
    if not user or not user.is_admin:
        raise HTTPException(403, "Admin only")

@app.post("/admin/models/{model_uuid}/lock")
def admin_lock_model(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); admin_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m: raise HTTPException(404, "Model not found")
    m.is_locked = True
    db.add(m); db.commit()
    return RedirectResponse(url=f"/models/{model_uuid}", status_code=302)

@app.post("/admin/models/{model_uuid}/unlock")
def admin_unlock_model(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); admin_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m: raise HTTPException(404, "Model not found")
    m.is_locked = False
    db.add(m); db.commit()
    return RedirectResponse(url=f"/models/{model_uuid}", status_code=302)

@app.post("/admin/models/{model_uuid}/certify")
def admin_certify_model(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); admin_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m: raise HTTPException(404, "Model not found")
    m.is_certified = True
    db.add(m); db.commit()
    return RedirectResponse(url=f"/models/{model_uuid}", status_code=302)

@app.post("/admin/models/{model_uuid}/decertify")
def admin_decertify_model(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); admin_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m: raise HTTPException(404, "Model not found")
    m.is_certified = False
    db.add(m); db.commit()
    return RedirectResponse(url=f"/models/{model_uuid}", status_code=302)

@app.post("/admin/models/{model_uuid}/delete")
def admin_delete_model(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); admin_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m: raise HTTPException(404, "Model not found")
    # 物理ファイル群を DL 不可の退避先へ移動してから DB から削除
    try:
        src_dir = model_dir(m.uuid)
        dst_dir = os.path.join(SAFE_STORAGE_DIR, m.uuid + "_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.isdir(src_dir):
            for root, dirs, files in os.walk(src_dir):
                for fn in files:
                    sp = os.path.join(root, fn)
                    try:
                        shutil.move(sp, os.path.join(dst_dir, fn))
                    except Exception:
                        pass
    except Exception:
        pass
    # レコードを完全削除
    db.delete(m); db.commit()
    return RedirectResponse(url="/", status_code=302)

# -------------------------
# 運営：DBバックアップ（SQLiteのみ簡易対応）
# -------------------------
def _sqlite_path_from_url(url: str) -> Optional[str]:
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "", 1)
    if url.startswith("sqlite:////"):
        return url.replace("sqlite:////", "/", 1)
    return None

@app.post("/admin/db/backup")
def admin_db_backup(request: Request):
    # セッションから判定（DB不要）
    # SQLiteファイルを BACKUP_DIR にコピー
    # 非SQLiteは簡易実装では未対応
    from starlette.requests import Request as StarReq
    req: StarReq = request  # type: ignore
    # 簡易に current_user を取り直す
    db = SessionLocal()
    try:
        user = get_current_user(request, db); admin_required(user)
    finally:
        db.close()
    path = _sqlite_path_from_url(DATABASE_URL)
    if not path or not os.path.exists(path):
        raise HTTPException(400, "Backup only supports SQLite in this build")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(BACKUP_DIR, f"app_{ts}.db")
    shutil.copy2(path, dst)
    return PlainTextResponse(f"Backed up to: {dst}")

# -----------------------------------
# いいね
# -----------------------------------
@app.post("/models/{model_uuid}/like")
def like_model(model_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    m = db.query(Model).filter(Model.uuid == model_uuid, Model.is_public == True).first()
    if not m:
        raise HTTPException(404, "Model not found")
    exists = db.query(Like).filter(Like.user_uuid == user.uuid, Like.model_uuid == m.uuid).first()
    if exists:
        db.delete(exists)
        m.likes_count = max(0, m.likes_count - 1)
    else:
        db.add(Like(user_uuid=user.uuid, model_uuid=m.uuid))
        m.likes_count += 1
    m.updated_at = datetime.now(timezone.utc)
    db.add(m); db.commit()
    return RedirectResponse(url=f"/models/{m.uuid}", status_code=302)

# -----------------------------------
# ユニット
# -----------------------------------
@app.get("/units/create", response_class=HTMLResponse)
def unit_create_form(request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    my_models = db.query(Model).filter(Model.user_uuid == user.uuid).order_by(Model.created_at.desc()).all()
    return templates.TemplateResponse("unit_create.html", {"request": request, "user": user, "my_models": my_models})

@app.post("/units/create")
def unit_create(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    icon: Optional[UploadFile] = File(None),
    db: OrmSession = Depends(get_db)
):
    user = get_current_user(request, db); login_required(user)
    u = Unit(owner_user_uuid=user.uuid, name=name, description=description)
    db.add(u); db.flush()  # uuid確定
    if icon:
        icon_name = f"{u.uuid}_icon{os.path.splitext(icon.filename or '')[1] or '.png'}"
        unit_dir = os.path.join(MODEL_STORAGE_DIR, "units", u.uuid)
        os.makedirs(unit_dir, exist_ok=True)
        u.icon_path = save_small_file(os.path.join(unit_dir, icon_name), icon)
    u.updated_at = datetime.now(timezone.utc)
    db.add(u); db.commit()
    return RedirectResponse(url=f"/units/{u.uuid}", status_code=302)

@app.get("/units/{unit_uuid}", response_class=HTMLResponse)
def unit_detail(unit_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    u = db.query(Unit).filter(Unit.uuid == unit_uuid).first()
    if not u:
        raise HTTPException(404, "Unit not found")
    user = get_current_user(request, db)
    return templates.TemplateResponse("unit_detail.html", {"request": request, "user": user, "unit": u})

@app.get("/units/{unit_uuid}/edit", response_class=HTMLResponse)
def unit_edit_form(unit_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    u = db.query(Unit).filter(Unit.uuid == unit_uuid).first()
    if not u or u.owner_user_uuid != user.uuid:
        raise HTTPException(403, "Forbidden")
    return templates.TemplateResponse("unit_edit.html", {"request": request, "user": user, "unit": u})

@app.post("/units/{unit_uuid}/edit")
def unit_edit(
    unit_uuid: str, request: Request,
    # 空欄は据え置きに
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    icon: Optional[UploadFile] = File(None),
    db: OrmSession = Depends(get_db)
):
    user = get_current_user(request, db); login_required(user)
    u = db.query(Unit).filter(Unit.uuid == unit_uuid).first()
    if not u or u.owner_user_uuid != user.uuid:
        raise HTTPException(403, "Forbidden")
    # 空欄なら変更しない
    if name is not None and name.strip() != "":
        u.name = name
    if description is not None and description.strip() != "":
        u.description = description
    if icon:
        unit_dir = os.path.join(MODEL_STORAGE_DIR, "units", u.uuid)
        os.makedirs(unit_dir, exist_ok=True)
        icon_name = f"{u.uuid}_icon{os.path.splitext(icon.filename or '')[1] or '.png'}"
        u.icon_path = save_small_file(os.path.join(unit_dir, icon_name), icon)
    u.updated_at = datetime.now(timezone.utc)
    db.add(u); db.commit()
    return RedirectResponse(url=f"/units/{u.uuid}", status_code=302)

@app.post("/units/{unit_uuid}/delete")
def unit_delete(unit_uuid: str, request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    u = db.query(Unit).filter(Unit.uuid == unit_uuid).first()
    if not u or u.owner_user_uuid != user.uuid:
        raise HTTPException(403, "Forbidden")
    # 関連だけ外せばファイルはアイコンのみ
    try:
        if u.icon_path and os.path.isfile(u.icon_path):
            os.remove(u.icon_path)
        unit_dir = os.path.join(MODEL_STORAGE_DIR, "units", u.uuid)
        if os.path.isdir(unit_dir):
            for root, dirs, files in os.walk(unit_dir, topdown=False):
                for f in files:
                    try: os.remove(os.path.join(root, f))
                    except: pass
                for d in dirs:
                    try: os.rmdir(os.path.join(root, d))
                    except: pass
            os.rmdir(unit_dir)
    except Exception:
        pass
    db.delete(u); db.commit()
    return RedirectResponse(url=f"/users/{user.uuid}", status_code=302)

@app.get("/media/units/{unit_uuid}/icon")
def media_unit_icon(unit_uuid: str, db: OrmSession = Depends(get_db)):
    u = db.query(Unit).filter(Unit.uuid == unit_uuid).first()
    if not u or not u.icon_path:
        raise HTTPException(404, "No unit icon")
    headers = {"Content-Disposition": build_content_disposition(os.path.basename(u.icon_path), inline=True)}
    return StreamingResponse(open(u.icon_path, "rb"), media_type=guess_mime(u.icon_path, "image/png"), headers=headers)

@app.post("/units/{unit_uuid}/add_model")
def unit_add_model(unit_uuid: str, model_uuid: str = Form(...), request: Request = None, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    u = db.query(Unit).filter(Unit.uuid == unit_uuid).first()
    if not u or u.owner_user_uuid != user.uuid:
        raise HTTPException(403, "Forbidden")
    m = db.query(Model).filter(Model.uuid == model_uuid).first()
    if not m or m.user_uuid != user.uuid:
        raise HTTPException(400, "Invalid model")
    if m not in u.models:
        u.models.append(m); u.updated_at = datetime.now(timezone.utc)
        db.add(u); db.commit()
    return RedirectResponse(url=f"/units/{unit_uuid}", status_code=302)

# -----------------------------------
# 管理者：お知らせ
# -----------------------------------
@app.get("/admin/announcements", response_class=HTMLResponse)
def admin_announcements(request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    if not user.is_admin:
        raise HTTPException(403, "Admin only")
    ann = db.query(Announcement).order_by(Announcement.created_at.desc()).all()
    return templates.TemplateResponse("admin_announcements.html", {"request": request, "user": user, "announcements": ann})

@app.post("/admin/announcements")
def admin_create_announcement(
    request: Request,
    title: str = Form(...),
    content: str = Form(""),
    db: OrmSession = Depends(get_db)
):
    user = get_current_user(request, db); login_required(user)
    if not user.is_admin:
        raise HTTPException(403, "Admin only")
    a = Announcement(admin_user_uuid=user.uuid, title=title, content=content)
    db.add(a); db.commit()
    return RedirectResponse(url="/admin/announcements", status_code=302)

@app.post("/admin/announcements/{announcement_uuid}/delete")
def admin_delete_announcement(
    announcement_uuid: str,
    request: Request,
    db: OrmSession = Depends(get_db)
):
    user = get_current_user(request, db); 
    # 運営のみ
    if not user or not user.is_admin:
        raise HTTPException(403, "Admin only")
    a = db.query(Announcement).filter(Announcement.uuid == announcement_uuid).first()
    if not a:
        raise HTTPException(404, "Announcement not found")
    db.delete(a); db.commit()
    return RedirectResponse(url="/admin/announcements", status_code=302)

# =========================
# パスワード変更（本人のみ）
# =========================
@app.get("/users/me/password", response_class=HTMLResponse)
def password_change_form(request: Request, db: OrmSession = Depends(get_db)):
    user = get_current_user(request, db); login_required(user)
    return templates.TemplateResponse("password_change.html", {"request": request, "user": user})

@app.post("/users/me/password")
def password_change(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    new_password_confirm: str = Form(...),
    db: OrmSession = Depends(get_db)
):
    user = get_current_user(request, db); login_required(user)
    # 現在のパスワード確認
    if not bcrypt.verify(current_password, user.password_hash):
        raise HTTPException(400, "現在のパスワードが正しくありません")
    # 新パスワード検証
    if new_password != new_password_confirm:
        raise HTTPException(400, "新しいパスワードが一致しません")
    if len(new_password) < 8:
        raise HTTPException(400, "新しいパスワードは8文字以上にしてください")
    # 変更
    user.password_hash = bcrypt.hash(new_password)
    db.add(user); db.commit()
    return RedirectResponse(url=f"/users/{user.uuid}", status_code=302)

# -----------------------------------
# 検索（/search）
# -----------------------------------
@app.get("/search", response_class=HTMLResponse)
def search_page(
    request: Request,
    q: Optional[str] = None,
    month: Optional[str] = None,  # "YYYY-MM"
    license: Optional[str] = None,  # ライセンス名
    db: OrmSession = Depends(get_db)
):
    query = db.query(Model).filter(Model.is_public == True)

    # キーワード
    if q:
        esc = escape_like(q.strip())
        like_expr = f"%{esc}%"
        query = query.filter(
            (Model.name.ilike(like_expr, escape='\\')) |
            (Model.description.ilike(like_expr, escape='\\'))
        )

    # 更新月
    if month:
        try:
            y, m = month.split("-")
            y = int(y); m = int(m)
            start = datetime(y, m, 1, tzinfo=timezone.utc)
            if m == 12:
                end = datetime(y+1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(y, m+1, 1, tzinfo=timezone.utc)
            query = query.filter(Model.updated_at >= start, Model.updated_at < end)
        except Exception:
            pass

    # ライセンス（テンプレ名）
    if license:
        query = query.filter(Model.license_name == license)

    results = query.order_by(Model.updated_at.desc()).limit(100).all()
    user = get_current_user(request, db)
    return templates.TemplateResponse("search.html", {
        "request": request, "user": user, "results": results,
        "q": q or "", "month": month or "", "license_selected": license or "",
        "license_templates": LICENSE_TEMPLATES
    })

# -----------------------------------
# 利用規約
# -----------------------------------
@app.get("/terms", response_class=HTMLResponse)
def terms_page(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

# -----------------------------------
# Every1KoeSpeech 互換API（最小）
# -----------------------------------
class HubSpeakerStyle(BaseModel):
    local_id: int
    name: Optional[str] = None

class HubSpeaker(BaseModel):
    speaker_uuid: str
    name: Optional[str] = None
    styles: List[HubSpeakerStyle] = []

class HubModelSummary(BaseModel):
    model_uuid: str
    title: str
    author: Optional[str] = None
    license: Optional[str] = None
    architecture: Optional[str] = None
    format: Optional[str] = "ONNX"
    download_count: int = 0
    updated_at: Optional[str] = None
    speakers: List[HubSpeaker] = []
    detail_url: str

class HubSearchResponse(BaseModel):
    items: List[HubModelSummary]
    total: int
    page: int
    page_size: int

@app.get("/hub/search", response_model=HubSearchResponse)
def hub_search(q: Optional[str] = None, sort: str = "download_count_desc", page: int = 1, page_size: int = 20, db: OrmSession = Depends(get_db)):
    query = db.query(Model).filter(Model.is_public == True)
    if q:
        like = f"%{escape_like(q)}%"
        query = query.filter((Model.name.ilike(like, escape='\\')) | (Model.description.ilike(like, escape='\\')))
    if sort == "download_count_desc":
        query = query.order_by(Model.downloads.desc(), Model.created_at.desc())
    elif sort == "likes_desc":
        query = query.order_by(Model.likes_count.desc(), Model.created_at.desc())
    elif sort == "new_desc":
        query = query.order_by(Model.created_at.desc())
    total = query.count()
    items_db = query.offset((page - 1) * page_size).limit(page_size).all()

    items = []
    for m in items_db:
        author_name = m.author.username if m.author else "unknown"
        items.append(HubModelSummary(
            model_uuid=m.uuid,
            title=m.name,
            author=author_name,
            license=m.license_name,
            architecture=None,
            format="ONNX",
            download_count=m.downloads,
            updated_at=m.updated_at.replace(tzinfo=timezone.utc).isoformat(),
            speakers=[],
            detail_url=f"/hub/models/{m.uuid}",
        ))
    return HubSearchResponse(items=items, total=total, page=page, page_size=page_size)

@app.get("/hub/models/{model_uuid}")
def hub_model_detail(model_uuid: str, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid, Model.is_public == True).first()
    if not m:
        raise HTTPException(404, "Model not found")

    return {
         "model_uuid": m.uuid,
         "title": m.name,
         "description": m.description,
         "version": "1.0.0",
         "manifest_version": "1.0",
         "license": m.license_name,
         "architecture": None,
         "format": "ONNX",
         "languages": ["ja"],
         "speakers": [],
         "file": {
             "kind": "AIVMX",
             "size_bytes": m.size_bytes,
             "sha256": m.sha256,
             "download_url": f"/assets/models/{m.uuid}/model.aivmx"
         },
         "stats": {
             "download_count": m.downloads,
             "like_count": m.likes_count
         },
        "locked": m.is_locked,
        "certified": m.is_certified,
         "published_at": (m.created_at or datetime.now(timezone.utc)).replace(tzinfo=timezone.utc).isoformat(),
         "updated_at": (m.updated_at or m.created_at).replace(tzinfo=timezone.utc).isoformat()
     }

@app.get("/assets/models/{model_uuid}/model.aivmx")
def hub_model_download(model_uuid: str, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid, Model.is_public == True).first()
    if not m:
        raise HTTPException(404, "Model not found")
    # 施錠中はAPI経由のDLも不可
    if m.is_locked:
        # Every1KoeSpeechクライアントからの取得も拒否（人間向けDLとレスポンス方針を統一）
        raise HTTPException(status_code=403, detail="Locked model: download disabled")
    m.downloads += 1
    m.updated_at = datetime.now(timezone.utc)
    db.add(m); db.commit()

    filename = os.path.basename(m.file_path)
    headers = {"Content-Disposition": build_content_disposition(filename, inline=False)}
    return Stream

@app.get("/v1/aivm-models/{model_uuid}/download")
def hub_model_download(model_uuid: str, db: OrmSession = Depends(get_db)):
    m = db.query(Model).filter(Model.uuid == model_uuid, Model.is_public == True).first()
    if not m:
        raise HTTPException(404, "Model not found")
    # 施錠中はAPI経由のDLも不可
    if m.is_locked:
        # Every1KoeSpeechクライアントからの取得も拒否（人間向けDLとレスポンス方針を統一）
        raise HTTPException(status_code=403, detail="Locked model: download disabled")
    m.downloads += 1
    m.updated_at = datetime.now(timezone.utc)
    db.add(m); db.commit()

    filename = os.path.basename(m.file_path)
    headers = {"Content-Disposition": build_content_disposition(filename, inline=False)}
    return StreamingResponse(open(m.file_path, "rb"), media_type="application/octet-stream", headers=headers)

# -----------------------------------
# メイン
# -----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8012, reload=True)
