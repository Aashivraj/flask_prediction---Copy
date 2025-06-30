from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

def create_mysql_connection():
    engine = create_engine(
        "mysql+pymysql://limer_ai:GsbldqgCPgnI8N4y@3.108.183.60:3306/lime_pos_production2?charset=utf8mb4",
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return engine

def check_connection():
    try:
        engine = create_mysql_connection()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("connected")
            for row in result:
                print("Test query result:", row)
    except SQLAlchemyError as e:
        print("not connected")
        print("Error:", e)

if __name__ == "__main__":
    check_connection()
