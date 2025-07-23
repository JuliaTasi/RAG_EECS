import json
import string
import os
from typing import List, Dict
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QAPair:
    """Represents a question-answer pair for evaluation"""
    id: str
    question: str
    answer: str
    url: str

def normalize_answer(answer: str) -> str:
    """Normalize answer for evaluation (lowercase, remove punctuation, extra spaces)"""
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = ''.join(char for char in answer if char not in string.punctuation)
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer

def exact_match_score(predicted: str, reference: str) -> int:
    """Calculate exact match score (1 if exact match, 0 otherwise)"""
    pred_normalized = normalize_answer(predicted)
    ref_normalized = normalize_answer(reference)
    
    return 1 if pred_normalized == ref_normalized else 0

def load_qa_pairs(filepath: str) -> List[QAPair]:
    """Load QA pairs from JSONL file (one JSON object per line)"""
    qa_pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            item = json.loads(line)
            
            # Generate ID if not present
            qa_id = item.get('id', f"{line_num:03d}")
            
            qa_pair = QAPair(
                id=qa_id,
                question=item['question'],
                answer=item['answer'],
                url=item['url']
            )
            qa_pairs.append(qa_pair)
    
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {filepath}")
    return qa_pairs

def evaluate_rag_system(rag_system, qa_pairs: List[QAPair], top_k: int = 3) -> Dict:
    """Evaluate RAG system on QA pairs using exact match"""
    logger.info(f"Starting evaluation of {len(qa_pairs)} QA pairs...")
    
    results = []
    correct_answers = 0
    total_questions = len(qa_pairs)
    
    for i, qa_pair in enumerate(qa_pairs):
        logger.info(f"Processing question {i+1}/{total_questions}: {qa_pair.id}")
        
        try:
            # Get answer from RAG system
            rag_result = rag_system.answer_question(qa_pair.question, top_k)
            predicted_answer = rag_result['answer']
            
            # Calculate exact match
            em_score = exact_match_score(predicted_answer, qa_pair.answer)
            correct_answers += em_score
            
            # Store individual result
            result = {
                'id': qa_pair.id,
                'question': qa_pair.question,
                'reference_answer': qa_pair.answer,
                'predicted_answer': predicted_answer,
                'exact_match': em_score,
                'url': qa_pair.url,
                'sources_found': len(rag_result['sources'])
            }
            results.append(result)
            
            # Print progress
            status = "✓" if em_score == 1 else "✗"
            logger.info(f"  {status} EM: {em_score} | Ref: '{qa_pair.answer}' | Pred: '{predicted_answer}'")
            
        except Exception as e:
            logger.error(f"Error processing question {qa_pair.id}: {str(e)}")
            result = {
                'id': qa_pair.id,
                'question': qa_pair.question,
                'reference_answer': qa_pair.answer,
                'predicted_answer': "",
                'exact_match': 0,
                'url': qa_pair.url,
                'sources_found': 0,
                'error': str(e)
            }
            results.append(result)
    
    # Calculate final metrics
    exact_match_accuracy = correct_answers / total_questions
    
    evaluation_results = {
        'exact_match_accuracy': exact_match_accuracy,
        'correct_answers': correct_answers,
        'total_questions': total_questions,
        'individual_results': results
    }
    
    logger.info(f"Evaluation completed!")
    logger.info(f"Exact Match Accuracy: {exact_match_accuracy:.3f} ({correct_answers}/{total_questions})")
    
    return evaluation_results

def print_results(results: Dict):
    """Print evaluation results in a clear format"""
    print("\n" + "="*80)
    print("RAG SYSTEM EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Exact Match Accuracy: {results['exact_match_accuracy']:.3f}")
    print(f"  Correct Answers: {results['correct_answers']}")
    print(f"  Total Questions: {results['total_questions']}")
    
    # Show correct answers
    correct_results = [r for r in results['individual_results'] if r['exact_match'] == 1]
    if correct_results:
        print(f"\nCORRECT ANSWERS ({len(correct_results)}):")
        for result in correct_results:
            print(f"  ✓ [{result['id']}] {result['question'][:60]}...")
            print(f"    Answer: {result['reference_answer']}")
    
    # Show incorrect answers
    incorrect_results = [r for r in results['individual_results'] if r['exact_match'] == 0]
    if incorrect_results:
        print(f"\nINCORRECT ANSWERS ({len(incorrect_results)}):")
        for result in incorrect_results:
            print(f"  ✗ [{result['id']}] {result['question'][:60]}...")
            print(f"    Expected: '{result['reference_answer']}'")
            print(f"    Got:      '{result['predicted_answer']}'")
            print(f"    URL:      {result['url']}")
    
    # Show errors if any
    error_results = [r for r in results['individual_results'] if 'error' in r]
    if error_results:
        print(f"\nERRORS ({len(error_results)}):")
        for result in error_results:
            print(f"  ⚠ [{result['id']}] {result['error']}")

def save_results(results: Dict, output_file: str):
    """Save results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")

def create_sample_qa_file():
    """Create a sample QA pairs file for testing"""
    sample_qa_pairs = [
        {
            "id": "001",
            "question": "What does EECS stand for at UC Berkeley?",
            "answer": "Electrical Engineering and Computer Sciences",
            "url": "https://eecs.berkeley.edu"
        },
        {
            "id": "002", 
            "question": "What minor options does Berkeley EECS offer for students pursuing other majors?",
            "answer": "EECS minor, CS Minor, and EIS Minor",
            "url": "https://eecs.berkeley.edu/academics/undergraduate/cs-ba"
        },
        {
            "id": "003",
            "question": "Who received the 2021 EECS Distinguished Alumni Award in Computer Science for groundbreaking work in cryptography?",
            "answer": "Ralph Merkle",
            "url": "https://eecs.berkeley.edu/2021/03/2021-distinguished-alumni"
        },
        {
            "id": "004",
            "question": "Which CS course at Berkeley was named one of the '5 Best CS Classes in the US'?",
            "answer": "CS61A",
            "url": "https://eecs.berkeley.edu/academics"
        },
        {
            "id": "005",
            "question": "Who was the first woman to earn a PhD in electrical engineering at UC Berkeley?",
            "answer": "Kawthar Zaki",
            "url": "https://eecs.berkeley.edu/about/history/a-relatively-recent-history-women-doctoral-graduates-in-electrical-engineering-and-computer-sciences-1969-1981"
        },
    ]
    
    with open('sample_qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(sample_qa_pairs, f, indent=2, ensure_ascii=False)
    
    print("Created sample_qa_pairs.json with 3 sample questions")
    return 'sample_qa_pairs.json'

def run_ablation_study(rag_system, qa_pairs: List[QAPair]):
    """Run ablation study with different top_k values"""
    print("\n" + "="*80)
    print("ABLATION STUDY - Different top_k values")
    print("="*80)
    
    top_k_values = [1, 3, 5, 7]
    ablation_results = []
    
    for top_k in top_k_values:
        print(f"\nTesting with top_k={top_k}...")
        results = evaluate_rag_system(rag_system, qa_pairs, top_k=top_k)
        accuracy = results['exact_match_accuracy']
        
        ablation_results.append({
            'top_k': top_k,
            'exact_match_accuracy': accuracy,
            'correct_answers': results['correct_answers']
        })
        
        print(f"  Exact Match Accuracy: {accuracy:.3f}")
    
    # Find best top_k
    best_result = max(ablation_results, key=lambda x: x['exact_match_accuracy'])
    print(f"\nBest performance: top_k={best_result['top_k']} with accuracy={best_result['exact_match_accuracy']:.3f}")
    
    return ablation_results