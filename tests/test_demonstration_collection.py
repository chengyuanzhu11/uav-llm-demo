"""Seed-collection boundaries; execution/oracle tests live in their own suites."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.collect_demonstrations import DemonstrationAgent, corrupt_candidate, load_standards
from uav_llm_demo.agents import PlannerProviderError

class DemonstrationCollectionTests(unittest.TestCase):
    def test_standards_are_explicit_agent_authored_pilot_not_evaluation(self):
        data=load_standards()
        self.assertEqual(24,len(data['cases']))
        self.assertEqual(6,len({c['standard']['family'] for c in data['cases']}))
        self.assertEqual(6,sum(bool(c['demonstration']['repair']) for c in data['cases']))
        for case in data['cases']:
            self.assertEqual('pilot',case['standard']['split'])
            self.assertFalse(case['standard']['human_reviewed'])
        bad=deepcopy(data);bad['cases'][0]['standard'].pop('intent')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cases.json';p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load_standards(p)

    def test_teacher_submits_only_actual_successful_preview_id(self):
        case=load_standards()['cases'][0]
        agent=DemonstrationAgent(case)
        self.assertEqual('query_geometry',json.loads(agent._request_json_content([], 'semantic task plan'))['tool'])
        self.assertEqual('preview_plan',json.loads(agent._request_json_content([], 'semantic task plan'))['tool'])
        messages=[dict(role='user',content='Tool observation:\n'+json.dumps(dict(ok=True,candidate_id='observed-37')))]
        value=json.loads(agent._request_json_content(messages,'semantic task plan'))
        self.assertEqual('observed-37',value['arguments']['candidate_id'])
        broken=DemonstrationAgent(case)
        broken._request_json_content([],'semantic task plan');broken._request_json_content([],'semantic task plan')
        with self.assertRaises(PlannerProviderError): broken._request_json_content([], 'semantic task plan')

    def test_repair_retains_rejected_context_without_mutating_valid_label(self):
        case=load_standards()['cases'][1];original=deepcopy(case)
        agent=DemonstrationAgent(case)
        agent._request_json_content([], 'semantic task plan')
        bad=json.loads(agent._request_json_content([], 'semantic task plan'))
        self.assertEqual(1,len(bad['arguments']['plan']['steps']))
        messages=[dict(role='user',content='Tool observation:\n'+json.dumps(dict(ok=False,diagnostic={'code':'missing_requirement'})))]
        good=json.loads(agent._request_json_content(messages, 'semantic task plan'))
        self.assertEqual(2,len(good['arguments']['plan']['steps']))
        self.assertEqual(original,case)

if __name__=='__main__': unittest.main()
