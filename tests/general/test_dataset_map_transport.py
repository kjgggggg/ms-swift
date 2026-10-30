"""Tests for sending the preprocessing function to the `datasets.map(num_proc=...)` workers.

`multiprocess` pickles the mapped function once per worker job with dill (a pure-Python pickler). The
function holds the template and its tokenizer, so it is pickled once with the C pickler and the same bytes
are sent to every job; functions that only dill can pickle keep the dill path.
"""
import json
import os
import tempfile
import unittest
from datasets import Dataset
from datasets.fingerprint import Hasher
from multiprocess import get_context
from multiprocess.reduction import ForkingPickler
from unittest import mock

from swift.dataset import RowPreprocessor
from swift.dataset.preprocessor import core
from swift.dataset.preprocessor.core import _MapFunction

SAMPLES = [{
    'messages': [{
        'role': 'user',
        'content': f'question {i}'
    }, {
        'role': 'assistant',
        'content': f'answer {i}'
    }]
} for i in range(8)]


class SuffixPreprocessor(RowPreprocessor):

    def __init__(self, suffix='!'):
        super().__init__()
        self.suffix = suffix

    def preprocess(self, row):
        row['messages'][-1]['content'] += self.suffix
        return row


class LambdaPreprocessor(SuffixPreprocessor):
    """Only dill can pickle this preprocessor."""

    def __init__(self):
        super().__init__()
        self.get_suffix = lambda: '?'

    def preprocess(self, row):
        row['messages'][-1]['content'] += self.get_suffix()
        return row


def _batched(preprocessor):
    return preprocessor.batched_preprocess


IS_FORK = get_context().get_start_method() == 'fork'


class TestMapFunctionTransport(unittest.TestCase):

    @unittest.skipUnless(IS_FORK, 'The C pickler is only used with the fork start method.')
    def test_pickle_once_with_c_pickler(self):
        map_func = _MapFunction(_batched(SuffixPreprocessor()), 2)
        with mock.patch.object(core.pickle, 'dumps', wraps=core.pickle.dumps) as dumps:
            payloads = [ForkingPickler.dumps(map_func) for _ in range(3)]
        self.assertEqual(dumps.call_count, 1)
        self.assertEqual(len(set(bytes(p) for p in payloads)), 1)
        loaded = ForkingPickler.loads(payloads[0])
        self.assertIsInstance(loaded, _MapFunction)
        self.assertEqual(loaded.num_proc, 2)
        self.assertEqual(loaded.func.__self__.suffix, '!')

    def test_dill_fallback(self):
        map_func = _MapFunction(_batched(LambdaPreprocessor()), 2)
        loaded = ForkingPickler.loads(ForkingPickler.dumps(map_func))
        self.assertIsInstance(loaded, _MapFunction)
        self.assertEqual(loaded.func.__self__.get_suffix(), '?')

    def test_fingerprint_ignores_transport(self):
        map_func = _MapFunction(_batched(SuffixPreprocessor()), 2)
        fingerprint = Hasher.hash(map_func)
        ForkingPickler.dumps(map_func)
        self.assertEqual(Hasher.hash(map_func), fingerprint)
        self.assertEqual(Hasher.hash(ForkingPickler.loads(ForkingPickler.dumps(map_func))), fingerprint)
        self.assertNotEqual(Hasher.hash(_MapFunction(_batched(SuffixPreprocessor('?')), 2)), fingerprint)


class TestMapWithWorkers(unittest.TestCase):

    def setUp(self):
        # In-memory datasets are mapped into `MODELSCOPE_CACHE`; keep every test isolated.
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        patcher = mock.patch.dict(os.environ, {'MODELSCOPE_CACHE': temp_dir.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _map(self, preprocessor, num_proc):
        dataset = Dataset.from_list(SAMPLES)
        dataset = preprocessor(dataset, num_proc=num_proc, batch_size=2, load_from_cache_file=False)
        messages = [row['messages'][-1] for row in dataset]
        return [(json.loads(m) if isinstance(m, str) else m)['content'] for m in messages]

    def test_same_results_as_single_process(self):
        expected = [f'answer {i}!' for i in range(len(SAMPLES))]
        self.assertEqual(self._map(SuffixPreprocessor(), 1), expected)
        self.assertEqual(self._map(SuffixPreprocessor(), 2), expected)

    def test_dill_only_preprocessor(self):
        expected = [f'answer {i}?' for i in range(len(SAMPLES))]
        self.assertEqual(self._map(LambdaPreprocessor(), 2), expected)


if __name__ == '__main__':
    unittest.main()
