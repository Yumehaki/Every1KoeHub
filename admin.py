# adminify.py
import os
import argparse
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import Base, User, DATABASE_URL, connect_args  # app.pyと同じ設定を利用

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", help="target user email")
    parser.add_argument("--username", help="target username")
    args = parser.parse_args()

    if not args.email and not args.username:
        print("Specify --email or --username")
        return

    engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    db = SessionLocal()

    try:
        q = db.query(User)
        if args.email:
            q = q.filter(User.email == args.email)
        if args.username:
            q = q.filter(User.username == args.username)
        u = q.first()
        if not u:
            print("User not found")
            return
        u.is_admin = True
        db.add(u); db.commit()
        print(f"Promoted {u.username} to admin.")
    finally:
        db.close()

if __name__ == "__main__":
    main()
