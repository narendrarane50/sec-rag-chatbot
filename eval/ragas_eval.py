"""
RAGAS Evaluation for SEC Analyst Models
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset

from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class SECAnalystEvaluator:
    def __init__(self):
        self.base_model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.finetuned_model_name = "narendrarane50/sec-analyst-phi3"
        self.test_data_path = "finetuning/data/test.jsonl"
        self.output_dir = Path("eval/results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_test_data(self, max_samples=50):
        """Load test dataset"""
        data = []
        with open(self.test_data_path) as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    'question': item['instruction'],
                    'context': item['input'],
                    'ground_truth': item['output']
                })
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"✓ Loaded {len(data)} test samples")
        return data
    
    def setup_models(self):
        """Load base and fine-tuned models"""
        print("\nLoading models...")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Force CPU for stability on Mac
        device = "cpu"
        print(f"  - Using device: {device}")
        
        # Base model
        print("  - Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
        )
        self.base_model = self.base_model.to(device)
        self.base_model.eval()
        
        # Fine-tuned model
        print("  - Loading fine-tuned model...")
        base_for_ft = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
        )
        
        from peft import PeftModel
        self.finetuned_model = PeftModel.from_pretrained(
            base_for_ft, 
            self.finetuned_model_name,
            device_map=None
        )
        self.finetuned_model = self.finetuned_model.to(device)
        self.finetuned_model.eval()
        
        print("✓ Models loaded!\n")
    
    def generate_answer(self, model, context, question):
        """Generate answer using model"""
        # Truncate context if too long
        if len(context) > 1000:
            context = context[:1000]
        
        input_text = f"Context:\n{context}"
        
        prompt = f"""<s>[INST] You are a financial analyst assistant. Use the provided SEC filing context to answer the question accurately.
    {input_text}
    {question} [/INST]"""
        
        print(f"\n[DEBUG] Generating answer for: {question[:50]}...")
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Tokenize with much shorter max length
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256  # Reduced from 512
        ).to(device)
        
        print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}, generating...")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=100,  # Reduced from 200
                    min_new_tokens=1,
                    temperature=0.7,
                    do_sample=False,  # Greedy decoding - faster
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            print(f"[DEBUG] Generation done!")
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("[/INST]")[-1].strip() if "[/INST]" in response else response.strip()
            
            return answer
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return "ERROR"
    
    def generate_predictions(self, test_data, model, system_name):
        """Generate predictions for all test samples"""
        predictions = []
        
        for item in tqdm(test_data, desc=f"Generating {system_name} predictions"):
            try:
                answer = self.generate_answer(model, item['context'], item['question'])
                predictions.append({
                    'question': item['question'],
                    'context': item['context'],
                    'answer': answer,
                    'ground_truth': item['ground_truth']
                })
            except Exception as e:
                print(f"Error: {e}")
                predictions.append({
                    'question': item['question'],
                    'context': item['context'],
                    'answer': "ERROR",
                    'ground_truth': item['ground_truth']
                })
        
        return predictions
    
    def prepare_ragas_dataset(self, predictions):
        """Convert predictions to RAGAS dataset format"""
        data = {
            'question': [p['question'] for p in predictions],
            'answer': [p['answer'] for p in predictions],
            'contexts': [[p['context']] for p in predictions],
            'ground_truth': [p['ground_truth'] for p in predictions]
        }
        return Dataset.from_dict(data)
    
    def run_evaluation(self, max_samples=50):
        """Run complete evaluation"""
        print("="*70)
        print("SEC ANALYST - RAGAS EVALUATION")
        print("="*70)
        
        # Load test data
        test_data = self.load_test_data(max_samples)
        
        # Setup models
        self.setup_models()
        
        # Generate predictions
        print("Generating predictions...\n")
        base_predictions = self.generate_predictions(test_data, self.base_model, "Base Model")
        finetuned_predictions = self.generate_predictions(test_data, self.finetuned_model, "Fine-tuned Model")
        
        # Save predictions
        pred_dir = self.output_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        
        pd.DataFrame(base_predictions).to_csv(pred_dir / "base_predictions.csv", index=False)
        pd.DataFrame(finetuned_predictions).to_csv(pred_dir / "finetuned_predictions.csv", index=False)
        print(f"\n✓ Predictions saved to {pred_dir}/\n")
        
        # Prepare datasets
        base_dataset = self.prepare_ragas_dataset(base_predictions)
        finetuned_dataset = self.prepare_ragas_dataset(finetuned_predictions)
        
        # Run RAGAS evaluation
        print("Running RAGAS metrics...\n")
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]
        
        print("Evaluating base model...")
        base_results = evaluate(base_dataset, metrics=metrics)
        
        print("Evaluating fine-tuned model...")
        finetuned_results = evaluate(finetuned_dataset, metrics=metrics)
        
        # Save results
        results = {
            'test_set_size': len(test_data),
            'base_model': {
                'name': self.base_model_name,
                'scores': dict(base_results)
            },
            'finetuned_model': {
                'name': self.finetuned_model_name,
                'scores': dict(finetuned_results)
            }
        }
        
        with open(self.output_dir / "ragas_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display results
        self.display_results(base_results, finetuned_results)
        
        print(f"\n✓ Results saved to {self.output_dir}/")
        
        return results
    
    def display_results(self, base_results, finetuned_results):
        """Display comparison results"""
        comparison = pd.DataFrame({
            'Metric': ['Faithfulness', 'Answer Relevancy', 'Context Precision', 
                      'Context Recall', 'Answer Correctness'],
            'Base Model': [
                base_results['faithfulness'],
                base_results['answer_relevancy'],
                base_results['context_precision'],
                base_results['context_recall'],
                base_results['answer_correctness'],
            ],
            'Fine-tuned': [
                finetuned_results['faithfulness'],
                finetuned_results['answer_relevancy'],
                finetuned_results['context_precision'],
                finetuned_results['context_recall'],
                finetuned_results['answer_correctness'],
            ]
        })
        
        # Calculate improvement
        comparison['Δ (%)'] = (
            (comparison['Fine-tuned'] - comparison['Base Model']) / comparison['Base Model'] * 100
        ).round(1)
        
        print("\n" + "="*70)
        print("RAGAS EVALUATION RESULTS")
        print("="*70)
        print(comparison.to_string(index=False))
        print("="*70)


if __name__ == "__main__":
    evaluator = SECAnalystEvaluator()
    results = evaluator.run_evaluation(max_samples=5)