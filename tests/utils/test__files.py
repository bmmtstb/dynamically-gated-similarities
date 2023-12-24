import os.path
import unittest

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidPathException
from dgs.utils.files import is_dir, is_file, is_project_dir, is_project_file, read_json, to_abspath
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
            (r"./test_data/866-200x300.jpg", False),  # missing tests folder
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_pd}"):
                self.assertEqual(is_project_dir(fp), is_pd)

    def test_is_file(self):
        for fp, is_af in [
            (r"./tests/utils/test__files.py", True),  # this files local path
            (r"./tests/test_data/866-200x300.jpg", True),
            (prepend_base(r"./tests/utils/test__files.py"), True),  # this files local path
            (prepend_base(r"./tests/test_data/866-200x300.jpg"), True),
            (prepend_base(r"./tests/test_data/"), False),  # directory not path
            (prepend_base(r"./test_data/866-200x300.jpg"), False),  # missing tests folder
            (r"", False),  # empty string
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_af}"):
                self.assertEqual(is_file(fp), is_af)

    def test_is_dir(self):
        for fp, is_ad in [
            (r"./tests/utils/test__files.py", False),  # this is a file
            (prepend_base(r"./tests/test_data/866-200x300.jpg"), False),  # absolute file
            (prepend_base(r"./tests/test_data/"), True),  # absolute directory
            (r"./tests/test_data/", True),  # local directory
            (prepend_base(r"./test_data/"), False),  # absolute but missing tests folder
            (r"", True),  # empty string is a valid local path
        ]:
            with self.subTest(msg=f"fp: {fp}, is: {is_ad}"):
                self.assertEqual(is_dir(fp), is_ad)

    def test_read_json(self):
        path = prepend_base(r"./tests/test_data/test.json")
        result = {"test": True}
        self.assertEqual(read_json(path), result)

    def test_read_json_invalid_path(self):
        path = prepend_base(r"./tests/test_data/faulty.json")
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
        ]:
            with self.subTest(msg=f"path: {path}, result: {result}"):
                if raises:
                    with self.assertRaises(InvalidPathException):
                        _ = to_abspath(path)
                else:
                    self.assertEqual(to_abspath(path), result)


if __name__ == "__main__":
    unittest.main()
