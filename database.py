from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

def create_mysql_connection():
    """Create and return a MySQL connection."""
    engine = create_engine(
        "mysql+pymysql://admin:+HOzu@A9D48MDCjF@3.108.183.60:8090/lime_pos_prediction2?charset=utf8mb4",
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return engine

def check_connection():
    try:
        engine = create_mysql_connection()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Database connection successful.")
            for row in result:
                print("Test query result:", row)
    except SQLAlchemyError as e:
        print("❌ Database connection failed.")
        print("Error:", e)

if __name__ == "__main__":
    check_connection()
