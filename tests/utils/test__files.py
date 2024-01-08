import os.path
import unittest

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidPathException
from dgs.utils.files import (
    is_abs_dir,
    is_abs_file,
    is_dir,
    is_file,
    is_project_dir,
    is_project_file,
    read_json,
    to_abspath,
)
from dgs.utils.types import FilePath


def prepend_base(path: FilePath) -> FilePath:
    """Prepend the base directory to the given path."""
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


class TestFiles(unittest.TestCase):
    def test_is_proj_file(self):
        for fp, is_pf in [
            (r"./tests/utils/test__files.py", True),  # this file
            (r"./tests/test_data/866-200x300.jpg", True),
            (r"./tests/test_data/866-201x301.jpg", False),  # wrong shape
            (r"./tests/test_data/", False),  # directory not path
            (r"", False),  # empty string
            (r"./test_data/866-200x300.jpg", False),  # missing tests folder
            (prepend_base(r"./tests/utils/test__files.py"), True),  # valid abspath which join somehow ignores
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_pf}"):
                self.assertEqual(is_project_file(fp), is_pf)

    def test_is_proj_dir(self):
        for fp, is_pd in [
            (r"./tests/utils/test__files.py", False),  # this file is a file not a dir
            (r"./tests/test_data/", True),  # valid directory
            (prepend_base(r"./tests/test_data/"), True),  # valid directory in abspath, which join ignores
            (prepend_base(r"./tests/dummy/"), False),  # invalid directory
            (r"", True),  # empty string is a valid local path
            (r".", True),  # this is a valid local path
            (r"./test_data/866-200x300.jpg", False),  # missing tests folder
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_pd}"):
                self.assertEqual(is_project_dir(fp), is_pd)

    def test_is_proj_file_w_root(self):
        tests_folder = os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/"))
        for fp, root, is_pf in [
            (r"./tests/utils/test__files.py", PROJECT_ROOT, True),  # this file
            (r"./test__files.py", os.path.dirname(os.path.realpath(__file__)), True),
            (r"./test_data/866-200x300.jpg", tests_folder, True),
            (r"./test_data/866-201x301.jpg", tests_folder, False),  # wrong shape
            (r"./test_data/", tests_folder, False),  # directory not path
            (r"", tests_folder, False),  # empty string and directory
            (r"./866-200x300.jpg", tests_folder, False),  # missing test_data folder
            (prepend_base(r"./tests/utils/test__files.py"), PROJECT_ROOT, True),  # valid abspath which join ignores
            (prepend_base(r"./tests/utils/test__files.py"), tests_folder, True),  # valid abspath which join ignores
            (r"./tests/utils/test__files.py", "/dummy/", False),
            (prepend_base(r"./tests/utils/test__files.py"), "dummy/dummy/", True),  # fixme: why is this true ??
            (prepend_base(r"./test__files.py"), tests_folder, False),  # invalid abspath due to prepend adding sth wrong
        ]:
            with self.subTest(msg=f"fp: {fp}, root: {root}, is: {is_pf}"):
                self.assertEqual(is_project_file(fp, root=root), is_pf)

    def test_is_proj_dir_w_root(self):
        tests_folder = os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/"))
        for fp, root, is_pd in [
            (r"./tests/utils/test__files.py", PROJECT_ROOT, False),  # this file is a file not a dir
            (r"./utils/test__files.py", tests_folder, False),  # this file is a file not a dir
            (r"./tests/test_data/", PROJECT_ROOT, True),  # valid directory
            (r"./test_data/", tests_folder, True),  # valid directory
            (prepend_base(r"./tests/test_data/"), PROJECT_ROOT, True),  # valid directory in abspath, which join ignores
            (prepend_base(r"./test_data/"), tests_folder, False),  # prepend made it to invalid path
            (prepend_base(r"./tests/dummy/"), PROJECT_ROOT, False),  # invalid directory
            (prepend_base(r"./dummy/"), tests_folder, False),  # invalid directory
            (r"", PROJECT_ROOT, True),  # empty string is a valid local path
            (r"", tests_folder, True),  # empty string is a valid local path
            (r".", PROJECT_ROOT, True),  # this is a valid local path
            (r".", tests_folder, True),  # this is a valid local path
            (r"./test_data/", PROJECT_ROOT, False),  # missing tests folder
        ]:
            with self.subTest(msg=f"fp: {fp}, root: {root}, is: {is_pd}"):
                self.assertEqual(is_project_dir(fp, root=root), is_pd)

    def test_is_abs_file(self):
        for fp, is_af in [
            (r"./tests/utils/test__files.py", False),  # this files local path
            (r"./tests/test_data/866-200x300.jpg", False),
            (prepend_base(r"./tests/utils/test__files.py"), True),  # this files local path
            (prepend_base(r"./tests/test_data/866-200x300.jpg"), True),
            (prepend_base(r"./tests/test_data/"), False),  # directory not path
            (prepend_base(r"./test_data/866-200x300.jpg"), False),  # missing tests folder
            (r"", False),  # empty string
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_af}"):
                self.assertEqual(is_abs_file(fp), is_af)

    def test_is_abs_dir(self):
        for fp, is_ad in [
            (r"./tests/utils/test__files.py", False),  # this is a file
            (r"./tests/test_data/", False),  # local directory
            (prepend_base(r"./tests/utils/test__files.py"), False),  # absolute file
            (prepend_base(r"./tests/test_data/"), True),  # absolute directory
            (prepend_base(r"./test_data/"), False),  # absolute but missing tests folder
            (r"", False),  # empty string is not an absolute path
            (r".", False),  # valid dir but not valid absolute path
            (prepend_base(r""), True),  # project root should be a valid absolute path
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_ad}"):
                self.assertEqual(is_abs_dir(fp), is_ad)

    def test_is_file(self):
        for fp, is_f in [
            (r"./tests/utils/test__files.py", True),  # this files local path
            (r"./tests/test_data/866-200x300.jpg", True),
            (prepend_base(r"./tests/utils/test__files.py"), True),  # this files local path
            (prepend_base(r"./tests/test_data/866-200x300.jpg"), True),
            (prepend_base(r"./tests/test_data/"), False),  # directory not path
            (prepend_base(r"./test_data/866-200x300.jpg"), False),  # missing tests folder
            (r"", False),  # empty string
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_f}"):
                self.assertEqual(is_file(fp), is_f)

    def test_is_dir(self):
        for fp, is_d in [
            (r"./tests/utils/test__files.py", False),  # this is a file
            (prepend_base(r"./tests/test_data/866-200x300.jpg"), False),  # absolute file
            (prepend_base(r"./tests/test_data/"), True),  # absolute directory
            (r"./tests/test_data/", True),  # local directory
            (prepend_base(r"./test_data/"), False),  # absolute but missing tests folder
            (r"", True),  # empty string is a valid local path
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_d}"):
                self.assertEqual(is_dir(fp), is_d)

    def test_read_json(self):
        path = prepend_base(r"./tests/test_data/test.json")
        result = {"test": True}
        self.assertEqual(read_json(path), result)

    def test_read_json_invalid_path(self):
        for path in [prepend_base(r"./tests/test_data/faulty.json"), prepend_base(r"./tests/test_data/json.faulty")]:
            with self.subTest(msg=f"path: {path}"):
                with self.assertRaises(InvalidPathException):
                    read_json(path)

    def test_to_abspath(self):
        for path, result, raises in [
            (r"./tests/utils/test__files.py", prepend_base(r"./tests/utils/test__files.py"), False),
            (prepend_base(r"./tests/utils/test__files.py"), prepend_base(r"./tests/utils/test__files.py"), False),
            (r"./tests/utils/", prepend_base(r"./tests/utils/"), False),
            (prepend_base(r"./tests/utils/"), prepend_base(r"./tests/utils/"), False),
            (r"./utils/test__files.py", ..., True),  # tests folder missing
            (prepend_base(r"./utils/test__files.py"), ..., True),  # tests folder missing
            (r"./utils/", ..., True),  # tests folder missing
            (prepend_base(r"./utils/"), ..., True),  # tests folder missing
            (r"./tests/utils/invalid.py", ..., True),  # invalid file
            (prepend_base(r"./tests/utils/invalid.py"), ..., True),  # invalid file
            (  # clean dirty path
                r"./tests/../tests/./utils/././test__files.py",
                prepend_base(r"./tests/utils/test__files.py"),
                False,
            ),
            (  # clean dirty path
                prepend_base(r"./tests/../tests/./utils/././test__files.py"),
                prepend_base(r"./tests/utils/test__files.py"),
                False,
            ),
        ]:
            with self.subTest(msg=f"path: {path}, result: {result}"):
                if raises:
                    with self.assertRaises(InvalidPathException):
                        _ = to_abspath(path)
                else:
                    self.assertEqual(to_abspath(path), result)


if __name__ == "__main__":
    unittest.main()
