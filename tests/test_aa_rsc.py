"""Flight extraction regressions; fixtures mirror both AA page schemas."""
import json
import unittest
from scrapers.artificial_analysis import ArtificialAnalysisScraper


def flight(chunk):
    return 'self.__next_f.push(' + json.dumps([1, chunk]) + ')'


class FlightTests(unittest.TestCase):
    def test_split_chunks_and_metadata_before_rich_rows(self):
        metadata = {'slug': 'model', 'name': 'Model', 'releaseDate': '2026-09-01'}
        rich = {'slug': 'model', 'name': 'Model "quoted" \\ path',
                'intelligenceIndex': 0, 'price1mInputTokens': 0,
                'medianOutputTokensPerSecond': 120, 'codingIndex': '$undefined'}
        payload = '1:' + json.dumps({'models': [metadata]})
        payload += '\n2:' + json.dumps({'models': [rich]})
        midpoint = len(payload) // 2
        rows = ArtificialAnalysisScraper._parse_rsc_models(
            [flight(payload[:midpoint]), flight(payload[midpoint:])])
        self.assertEqual(len(rows), 1)
        model = ArtificialAnalysisScraper()._transform_model(rows[0])
        self.assertEqual(model['release_date'], '2026-09-01')
        self.assertEqual(model['intelligence_index'], 0)
        self.assertEqual(model['price_1m_input'], 0)
        self.assertEqual(model['median_output_speed'], 120)
        self.assertEqual(model['name'], rich['name'])
        self.assertNotIn('coding_index', model['metrics'])

    def test_legacy_large_array_and_nested_undefined(self):
        rows = [{'slug': str(i), 'name': f'Model {i}', 'intelligence_index': i,
                 'timescaleData': {'median_output_speed': '$undefined'}}
                for i in range(650)]
        parsed = ArtificialAnalysisScraper._parse_rsc_models(
            [flight(json.dumps({'models': rows}))])
        self.assertEqual(len(parsed), 650)
        self.assertEqual(parsed[-1]['intelligence_index'], 649)
        self.assertIsNone(parsed[0]['timescaleData']['median_output_speed'])

    def test_metadata_only_and_malformed_payloads_are_not_rich(self):
        self.assertEqual(ArtificialAnalysisScraper._parse_rsc_models([
            'self.__next_f.push(broken)', flight('{"models": [{"name": "Selector"}]}'),
            flight('{"models": [broken]')]), [])
