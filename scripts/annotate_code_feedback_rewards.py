#!/usr/bin/env python3
"""Context-aware, multi-pass reward annotation for m-a-p/Code-Feedback."""
from __future__ import annotations
import argparse, json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from datasets import load_dataset

VERSION="2.0-contextual"; SHIFT=0.52
RX=lambda p,f=re.I: re.compile(p,f)
EXEC=RX(r"^\s*(?:execution result|output|result)\s*:")
ERROR=RX(r"traceback \(most recent call last\)|\b(?:syntaxerror|typeerror|valueerror|nameerror|attributeerror|indexerror|keyerror|importerror|modulenotfounderror|filenotfounderror|runtimeerror|overflowerror|zerodivisionerror|tclerror|linalgerror|syntax error|type error|import error|recursion error|runtime error|logical error)\b|\b(?:compilation|execution|tests?)\s+failed\b|\b(?:segmentation fault|core dumped)\b")
MISSING=RX(r"\b(?:no|there is no) (?:code|answer|solution|implementation|function|output)(?: has been| was)? (?:generated|provided|produced|returned)\b|\b(?:assistant|you|it|code|answer|solution|implementation|function) (?:hasn['’]?t|haven['’]?t|wasn['’]?t|isn['’]?t|didn['’]?t).{0,60}\b(?:generated|provided|produced|returned|included|corrected)\b|\b(?:please|assistant should) (?:attempt to )?(?:provide|generate|write|produce).{0,50}\b(?:code|answer|solution|implementation|function|query|schema)\b|\b(?:code|query|schema|implementation|answer|solution) is missing\b|\b(?:code|implementation|solution) (?:was|is) not (?:actually )?(?:written|provided|executable)\b|\bpseudo[- ]code.{0,50}not executable\b",re.I|re.S)
NEGATIVE=RX(r"\b(?:incorrect(?:ly)?|wrong|not correct|does not work|doesn['’]?t work|not working|not correctly|does not correctly|doesn['’]?t correctly|fails? to|failed to|not handling|does not handle|doesn['’]?t handle|logical error|bug|violates?|contradicts?|against the requirements?|not in line with (?:the )?(?:task )?requirements?|not in (?:the )?requested \w+|will not compile|misses out|should not \w+|only works for|only (?:checks|handles|generates|returns|supports|uses)|not \w+ correctly|isn['’]?t .{0,40} properly|originally requested|not as (?:originally )?requested|different from (?:the )?(?:request|requirement)|does not meet|doesn['’]?t meet|does not \w+|doesn['’]?t \w+|(?:throws?|throwing|has|contains?) (?:an? )?(?:syntax |runtime |logical )?error|not producing|unexpected result|not defined|unable to|cannot|can['’]?t|still (?:has|throws?|fails?|does not)|infinite loop|does not compile|doesn['’]?t compile)\b")
CORRECTIVE_EXPECTATION=RX(r"\b(?:do(?:es)? not perform as expected|are currently just|is currently just|should (?:actually )?(?:return|produce|compute|calculate|implement|play|cache|use|handle|include|be implemented)|should be (?:implemented|used|called|removed|replaced|returned)|not the (?:sum|product|output|result|value)|instead of)\b")
ASSISTANT_ADMISSION=RX(r"\b(?:i apologize for (?:the |any )?(?:misunderstanding|oversight|mistake|error|confusion)|you(?:'re| are) right|i see the issue|i understand the issue|my mistake|was erroneously|were erroneously|you are correct|thanks? for (?:pointing|highlighting)|thank you for (?:pointing|highlighting))\b")
HARD_FAIL=RX(r"\b(?:incorrect output|wrong output|not producing (?:the )?(?:correct|expected) output|not working|does not work|doesn['’]?t work|logical error|\bbug\b|against the requirements?|violates?|contradicts?|does not meet|doesn['’]?t meet|not correctly|does not correctly|incorrectly|not defined|does not compile|doesn['’]?t compile|infinite loop|only works for)\b")
CORRECTED=RX(r"\b(?:assistant|code|function|issue|bug|error|problem).{0,80}\b(?:was|has been|is|successfully)?\s*corrected\b|\b(?:now|finally)\s+(?:works?|passes?|produces? the correct)\b",re.I|re.S)
POSITIVE=RX(r"\b(?:works? well|works? correctly|correct output|correct result|successfully|well[- ]structured|well[- ]formatted|well[- ]written|well explained|implemented correctly|efficient and correct|good solution|great solution|excellent|no improvements needed|looks good)\b")
OPTIMIZE=RX(r"\b(?:optimi[sz]\w*|more efficient|not efficient|inefficient|efficiency|time complexity|space complexity|performance|scalab\w*|memory[- ]efficient|additional space|extra space|resource usage|unnecessary comparisons?|vectoriz\w*)\b")
STYLE=RX(r"\b(?:comments?|documentation|formatting|formatted|indentation|readability|clarity|clearer|pep\s*8|style guide|maintainability|variable names?)\b")
TRANSLATE=RX(r"\b(?:translate|transcode|rewrite|refactor|adaptation|version|compliant|work in|convert|switching to|turn this into)\b")
EXTEND=RX(r"\b(?:add|includ\w*|support|consider|ensure|enhanc\w*|extend|introduce|implement|modif\w*|revise|update|improv\w*|handle|remove|adjust|correct|check|treat|address|avoid|move on|would be better|in addition|lacks?|missing|validation|security|robustness|edge cases?|test\w*|explain|elaborate|demonstrate|provide|proceed|generat\w*)\b")
QUESTION=RX(r"\?|^\s*(?:what|why|how|when|where|which|can|could|would|is|are|do|does|should|may|i don['’]?t understand)\b")
FEEDBACK=RX(r"\b(?:the|this|your|previous|above|assistant['’]?s?)\s+(?:code|function|solution|answer|response|approach|algorithm|implementation|output|result)|\bthe assistant\b")
EVAL_OPEN=RX(r"^\s*(?:while )?(?:the|this|your|previous|above|assistant['’]?s?)\s+(?:code|function|solution|answer|response|approach|algorithm|implementation|output|result)|^\s*(?:although|however|there (?:is|seems)|it (?:does|doesn|fails|cannot|can['’]?t|lacks))\b")
PACKED_OPEN=RX(r"^\s*(?:i have this problem|i am faced with|given|you are (?:given|tasked)|write|create|implement|develop|design|construct|build|devise|craft|compose|engineer|count|code|complete|combine|debug|ascertain|produce|investigate|show|process|perform|propose|print|draft|detail|retrieve|conceptualize|determine|identify|calculate|find|sort|formulate|use an algorithm|there are `?n|an array|a path|strings? `)\b")
TASK_OPEN=RX(r"^\s*(?:write|create|implement|develop|design|construct|build|generate|given|you are|craft|devise|using|calculate|find|sort|transform|convert|rewrite|make|use|parse|elaborate|compose|engineer|count|code|complete|combine|debug|ascertain|produce|investigate|show|process|perform|propose|print|draft|detail|retrieve|conceptualize|determine|identify|return|output|edit|fix|refactor|formulate|suggest|compute|define|change|reverse|delve|elucidate|conduct|concoct|conceive|provide|provided|swap|architect|within|show me|help me|please (?:write|create|implement|construct|complete|help)|you need to|i (?:have|need|want|am|used|would like|['’]?m)|there (?:is|are)|an? |the set|strings? |in the context|dilemma)\b")
PROBLEM=RX(r"\b(?:description|example|input|constraints?|requirements?|task|problem|given an?|write a|implement an?|return the|def |class |```)\b")
SOFT=RX(r"\b(?:could be|consider|would be better|might|may)\b")
DISPROVES_EXECUTION=RX(r"\b(?:incorrect output|wrong output|not producing (?:the )?(?:correct|expected) output|does not produce (?:the )?(?:correct|expected) output|logical error|against the requirements?|violates? (?:the )?(?:requirements?|constraints?)|not correctly (?:calculating|computing|returning|handling)|fails? to handle .{0,50} correctly|bug in (?:the )?(?:return|calculation|logic)|should return .{0,80} not)\b",re.I|re.S)

def cos(a,b):
 d=float(np.linalg.norm(a)*np.linalg.norm(b)); return None if not d or not math.isfinite(d) else float(np.dot(a,b)/d)
def embeddings(path):
 if not path.exists(): return None
 with np.load(path) as z:return np.asarray(z['embeddings'],dtype=np.float32)
def negated_corrected(t):return bool(RX(r"\b(?:not|hasn['’]?t|haven['’]?t|wasn['’]?t|isn['’]?t|didn['’]?t).{0,50}\bcorrected\b",re.I|re.S).search(t))
def signals(msgs,i,emb):
 t=msgs[i]['content'].strip(); pi=i-2 if i>=2 else None; sim=cos(emb[i],emb[pi]) if emb is not None and pi is not None else None
 next_assistant=msgs[i+1]['content'] if i+1<len(msgs) and msgs[i+1]['role']=='assistant' else ''
 feedback=bool(FEEDBACK.search(t)); packed=bool(PACKED_OPEN.search(t)); eval_open=bool(EVAL_OPEN.search(t)); question=bool(QUESTION.search(t)); markers=len(PROBLEM.findall(t)); task=bool(TASK_OPEN.search(t))
 standalone=not eval_open and (task or markers>=2 or len(t)>=500 or (question and not feedback))
 corrective=bool(CORRECTIVE_EXPECTATION.search(t)); admission=bool(ASSISTANT_ADMISSION.search(next_assistant))
 return {'text':t,'packed_opening':packed,'execution':bool(EXEC.search(t)),'execution_error':bool(EXEC.search(t) and ERROR.search(t)),'missing':bool(MISSING.search(t)),'negative':bool(NEGATIVE.search(t) or corrective or (ERROR.search(t) and (feedback or eval_open))),'hard_failure':bool(HARD_FAIL.search(t) or (corrective and admission) or (ERROR.search(t) and (feedback or eval_open))),'corrective_expectation':corrective,'next_assistant_admission':admission,'feedback':feedback,'evaluation_opening':eval_open,'corrected':bool(CORRECTED.search(t)) and not negated_corrected(t),'positive':bool(POSITIVE.search(t)),'optimization':bool(OPTIMIZE.search(t)),'style':bool(STYLE.search(t)),'translation':bool(TRANSLATE.search(t)),'extension':bool(EXTEND.search(t)),'question':question,'standalone':standalone,'topic_similarity':sim,'topic_shift':sim is not None and sim<SHIFT,'problem_markers':markers}
def ann(r,e,reason,conf,rule,s,provisional=False,override=None):
 keys=('packed_opening','hard_failure','corrective_expectation','next_assistant_admission','execution','execution_error','missing','negative','feedback','evaluation_opening','corrected','positive','optimization','style','translation','extension','question','standalone','topic_similarity','topic_shift','problem_markers')
 return {'annotation_version':VERSION,'reward':r,'reward_event':e,'reward_reason':reason,'reward_confidence':conf,'reward_source':'rule','reward_rule':rule,'reward_evidence':{k:s[k] for k in keys if s.get(k) not in (False,None,0)},'reward_provisional':provisional,'contextual_override':override}
def classify(s,initial=False):
 if initial:return ann(0,'initial_request','The first user turn has no preceding assistant response.','certain','initial_user_turn',s)
 if s['execution_error']:return ann(-2,'execution_error','Execution contains a compiler, runtime, or test failure.','high','execution_failure',s)
 if s['missing']:return ann(-2,'missing_answer','The requested answer was not provided.','high','missing_answer',s)
 if s['hard_failure'] and not s['standalone']:return ann(-2,'correctness_issue','The user identifies a hard correctness, constraint, or functionality failure.','high','contextual_hard_failure',s)
 if s['execution']:return ann(2,'execution_success','Execution completed without an explicit failure.','medium','provisional_execution_success',s,True)
 if s['packed_opening'] and s['standalone']:return ann(1,'new_independent_task','The user introduces a complete standalone task.','high','explicit_standalone_task',s)
 if s['optimization']:return ann(1,'optimization_request','The user requests an efficiency or scalability improvement.','high','optimization_request',s)
 if s['style']:return ann(1,'style_request','The user requests readability, formatting, or documentation improvements.','high','style_request',s)
 if s['translation']:return ann(1,'translation_request','The user requests a language or platform translation.','high','translation_request',s)
 if s['extension']:return ann(1,'constructive_request','The user requests a constructive revision or extension.','medium','constructive_request',s)
 if s['negative'] and not s['standalone']:return ann(-2,'correctness_issue','The user identifies a correctness or functionality failure.','high','contextual_negative_evaluation',s)
 if s['corrected'] or s['positive']:return ann(2,'confirmed_success','The user explicitly confirms correctness or correction.','high','explicit_success_confirmation',s)
 if s['question']:return ann(1,'follow_up_question','The user asks a valid follow-up or standalone question.','medium','valid_question',s)
 if s['standalone']:return ann(1,'new_independent_task','The user introduces a complete actionable task.','medium','complete_task',s)
 return ann(None,'ambiguous','Contextual rules do not provide enough evidence.','low','unresolved_context',s)
def sequence_override(out,sigs):
 for i,m in enumerate(out):
  if m['role']!='user' or not m.get('reward_provisional'):continue
  ni=i+2
  if ni<len(out) and out[ni]['role']=='user':
   s=sigs[ni]
   if DISPROVES_EXECUTION.search(s['text']) and (s['feedback'] or s['evaluation_opening']) and not s['standalone']:
    m.update(reward=-2,reward_event='incorrect_execution_output',reward_reason='Later user feedback establishes that the executed output was incorrect.',reward_confidence='high',reward_rule='later_feedback_overrides_execution',reward_provisional=False,contextual_override={'type':'later_feedback','evidence_turn':ni+1,'previous_reward':2});continue
  m['reward_provisional']=False
def load_overrides(path):
 if not path.exists():return {}
 return {(int(x['id']),int(x['turn'])):x for x in json.loads(path.read_text())['overrides']}
def types(out):
 u=[m for m in out if m['role']=='user' and m['turn']>1]; t=[]
 if any(m.get('reward_event')=='new_independent_task' for m in u):t.append('single_turn_packing')
 if any(m.get('reward_event') in {'execution_success','execution_error','incorrect_execution_output'} for m in u):t.append('interaction_simulation')
 if any(m.get('reward_event') in {'execution_error','incorrect_execution_output'} for m in u):t.append('code_correction')
 if any(m.get('reward_event')=='optimization_request' for m in u):t.append('leetcode_style_optimization')
 return t or ['unclassified']
def annotate(row,idx,embdir,overrides):
 path=embdir/f'conv_{idx:06d}.npz'; emb=embeddings(path); out=[]; sigs={}; users=0
 for i,m in enumerate(row['messages']):
  o={'turn':i+1,'embedding_index':i,'role':m['role'],'content':m['content']}
  if m['role']=='user':s=signals(row['messages'],i,emb);sigs[i]=s;o.update(classify(s,users==0));users+=1
  out.append(o)
 sequence_override(out,sigs)
 for m in out:
  x=overrides.get((row['id'],m['turn']))
  if x:
   prev=m.get('reward');m.update(reward=x['reward'],reward_event=x['reward_event'],reward_reason=x['reason'],reward_confidence='certain',reward_source='manual_override',reward_rule='manual_override',reward_provisional=False,contextual_override={'type':'manual','previous_reward':prev})
 return {'annotation_version':VERSION,'id':row['id'],'dataset_index':idx,'embedding_path':str(path),'embedding_exists':path.exists(),'conversation_types':types(out),'messages':out}
def write_reviews(path,convs):
 selected=[];n=0
 for c in convs:
  u=[m for m in c['messages'] if m['role']=='user' and m.get('reward') is None]
  if u:n+=len(u);selected.append({**{k:c[k] for k in ('id','dataset_index','embedding_path','conversation_types')},'uncertain_turns':u,'messages':c['messages']})
 path.write_text(json.dumps({'annotation_version':VERSION,'dataset':'m-a-p/Code-Feedback','conversation_count':len(selected),'null_user_turn_count':n,'conversations':selected},ensure_ascii=False,indent=2)+'\n')
def write_audit(path,convs,n,seed):
 rng=random.Random(seed);b=defaultdict(list)
 for c in convs:
  for m in c['messages']:
   if m['role']=='user':b[m['reward_event']].append({'id':c['id'],'conversation_types':c['conversation_types'],'target_turn':m['turn'],'target_reward':m['reward'],'target_event':m['reward_event'],'messages':c['messages']})
 path.write_text(json.dumps({'annotation_version':VERSION,'per_event':n,'samples':{e:rng.sample(v,min(n,len(v))) for e,v in sorted(b.items())}},ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/med_data/code_feedback_rewards.jsonl'));p.add_argument('--summary',type=Path,default=Path('data/med_data/code_feedback_rewards.summary.json'));p.add_argument('--null-review',type=Path,default=Path('data/med_data/code_feedback_null_review.json'));p.add_argument('--audit',type=Path,default=Path('data/med_data/code_feedback_reward_audit.json'));p.add_argument('--overrides',type=Path,default=Path('data/med_data/code_feedback_reward_overrides.json'));p.add_argument('--embedding-dir',type=Path,default=Path('data/med_data/data/embeddings'));p.add_argument('--cache-dir',type=Path,default=Path('data/med_data/data/hf_cache'));p.add_argument('--max-samples',type=int);p.add_argument('--audit-per-event',type=int,default=20);a=p.parse_args()
 ds=load_dataset('m-a-p/Code-Feedback',split='train',cache_dir=str(a.cache_dir));ds=ds.select(range(min(a.max_samples,len(ds)))) if a.max_samples else ds
 for x in (a.output,a.summary,a.null_review,a.audit):x.parent.mkdir(parents=True,exist_ok=True)
 ov=load_overrides(a.overrides);convs=[annotate(r,i,a.embedding_dir,ov) for i,r in enumerate(ds)]
 with a.output.open('w') as f:
  for c in convs:f.write(json.dumps(c,ensure_ascii=False)+'\n')
 rc=Counter();ec=Counter();cc=Counter();rules=Counter();src=Counter();tc=Counter();over=0
 for c in convs:
  tc.update(c['conversation_types'])
  for m in c['messages']:
   if m['role']=='user':rc['null' if m['reward'] is None else str(m['reward'])]+=1;ec[m['reward_event']]+=1;cc[m['reward_confidence']]+=1;rules[m['reward_rule']]+=1;src[m['reward_source']]+=1;over+=m['contextual_override'] is not None
 write_reviews(a.null_review,convs);write_audit(a.audit,convs,a.audit_per_event,42)
 s={'annotation_version':VERSION,'dataset':'m-a-p/Code-Feedback','conversations':len(convs),'user_turns':sum(rc.values()),'reward_counts':dict(rc),'event_counts':dict(ec),'confidence_counts':dict(cc),'rule_counts':dict(rules),'source_counts':dict(src),'conversation_type_counts':dict(tc),'contextual_overrides':over,'manual_overrides_loaded':len(ov),'embeddings_found':sum(c['embedding_exists'] for c in convs),'embeddings_missing':sum(not c['embedding_exists'] for c in convs),'output':str(a.output),'null_review':str(a.null_review),'audit':str(a.audit)};a.summary.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
