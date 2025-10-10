# cursor description

import tempfile

import duckdb


def check_exception(f):
    had_exception = False
    try:
        f()
    except Exception:
        had_exception = True
    assert had_exception


class TestReadOnly:
    def test_readonly(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name

        # this is forbidden
        check_exception(lambda: duckdb.connect(":memory:", True))

        con_rw = duckdb.connect(db, False)
        con_rw.cursor().execute("create table a (i integer)")
        con_rw.cursor().execute("insert into a values (42)")
        con_rw.close()

        con_ro = duckdb.connect(db, True)
        con_ro.cursor().execute("select * from a").fetchall()
        check_exception(lambda: con_ro.execute("delete from a"))
        con_ro.close()

        con_rw = duckdb.connect(db, False)
        con_rw.cursor().execute("drop table a")
        con_rw.close()
