from project_kidra import db


def test_add_run() -> None:
    conn = db.connect()
    db.init_schema(conn)
    run_id = db.add_run(conn, "first")
    row = conn.execute("SELECT name FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["name"] == "first"
