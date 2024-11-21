import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import llama_cpp
import time
import numpy as np 

# Custom Medusa Head Implementation
class CustomMedusaHead(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # initialize head projections for token prediction
        self.head_projections = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, vocab_size) 
            for _ in range(num_heads)
        ])
        
    def forward(self, hidden_states):
        # Generate predictions from each head
        logits = []
        for head in self.head_projections:
            head_logits = head(hidden_states)
            logits.append(head_output)
        return torch.stack(logtis)
    
    # def generate_candidates(self, hidden_states, top_k=5):
    #     logits = self.forward(hidden_states)
    #     # select top-k candidates for each head
    #     candidates = []
    #     probabilities = []
    #     for head_logits in logits:
    #         probs = F.softmax(head_logits, dim=-1)
    #         topk = torch.topk(probs, k=top_k, dum=-1)
    #         candidates.append(topk.indices)
    #         probabilities.append(topk.values)
            
    #     return torch.stack(candidates), torch.stack(probabilities)

class OptimizedLLM:
    def __init__(self, model_path: str):
        # Model compilation with llama.cpp optimizations
        self.model = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=512,
            n_threads=8,
            seed=-1,
            f16_kv=True
        )
                
        # Custom Medusa implementation
        self.medusa = CustomMedusaHead(
            hidden_size=4096, # model's hidden size?
            num_heads=4,
            vocab_size=self.model.n_vocab()
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.medusa.to(self.device)
        
    def generate_with_speculative_decoding(self, prompt: str, max_tokens: int, temperature: float = 0.7):
        try:
            # Initial generation with base model
            base_completion = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            if not base_completion or 'choices' not in base_completion or not base_completion['choices']:
                return "Error: Base model generation failed"
                
            base_text = base_completion['choices'][0]['text']
            
            # Get the completion tokens
            completion_tokens = self.model.tokenize(bytes(base_text, 'utf-8'))
            
            if not completion_tokens:
                return base_text
                
            # Apply speculative decoding
            final_tokens = []
            current_tokens = completion_tokens[:1]  # Start with first token
            
            for i in range(1, len(completion_tokens)):
                # Get base model prediction for next token
                base_output = self.model.eval(current_tokens)
                
                if base_output is None:
                    break
                    
                # Convert logits to probabilities
                base_logits = torch.tensor(base_output['logits'][-1])
                base_probs = F.softmax(base_logits / temperature, dim=-1)
                
                # Get actual next token from completion
                actual_next_token = completion_tokens[i]
                
                # If probability of actual next token is high enough, keep it
                if base_probs[actual_next_token].item() > 0.1:
                    final_tokens.append(actual_next_token)
                else:
                    # Otherwise use base model's prediction
                    predicted_token = torch.argmax(base_probs).item()
                    final_tokens.append(predicted_token)
                
                current_tokens = final_tokens.copy()
                
                if len(final_tokens) >= max_tokens:
                    break
            
            # Detokenize final tokens
            try:
                final_text = self.model.detokenize(final_tokens).decode('utf-8')
                return final_text if final_text else base_text
            except Exception as e:
                print(f"Detokenization error: {e}")
                return base_text
                
        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback to basic generation
            completion = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            if completion and 'choices' in completion and completion['choices']:
                return completion['choices'][0]['text']
            return "Error: Generation failed"
         
        # for _ in range(max_tokens):
        #     base_output = self.model.eval(context)
        #     if base_output is None:
        #         break
        #     base_token = base_output['logits'][-1].argmax().item()
            
        #     # Get Medusa head predictions
        #     hidden_states = torch.tensor(base_output['hidden_states'][-1])
        #     medusa_candidates = self.medusa.generate_candidates(hidden_states, num_candidates=5)
            
        #     # Verify candidates
        #     verified_token = self._verify_candidates(base_token, medusa_candidates[0])
        #     generated_tokens.append(verified_token)
        #     context.append(verified_token)
            
        #     if verified_token == self.model.token_eos():
        #         break
        
        # return self.model.detokenize(generated_tokens).decode()
        
    # def generate_with_speculative_decoding(self, prompt: str, max_tokens: int):
    #     output = self.model(prompt, max_tokens=max_tokens)
    #     if output is None or 'choices' not in output or len(output['choices']) == 0:
    #         return "Error: No text generated"
    #     return output['choices'][0]['text']
        
        # base_output = self.model.generate(prompt, max_tokens)
        
        # # Speculative decoding with Medusa heads
        # hidden_states = self.model.get_hidden_states()
        # medusa_candidates = self.medusa(hidden_states)
        
        # # Select best candidates
        # final_output = self._verify_candidates(
        #     base_output, 
        #     medusa_candidates
        # )
        # return final_output
    
    
    # def _verify_candidates(self, base_token, candidates, probabilities, threshold=0.8):
    #     # If base token is in candidates and has high probability, use it
    #     if base_token in candidates:
    #         idx = np.where(candidates == base_token)[0][0]
    #         if probabilities[idx] > threshold:
    #             return base_token
                
    #     # Find highest probability candidate that meets threshold
    #     for token, prob in zip(candidates, probabilities):
    #         if prob > threshold:
    #             return int(token)
                
    #     # Fallback to base model prediction
    #     return base_token
    
    # def _verify_candidates(self, base_token, candidates):
        
    #     if base_token in candidates:
    #         return base_token
    #     else:
    #         return base_token
        
    #     # verified_tokens = []
    #     # for base, specs in zip(base_output, candidates):
    #     #     # Choose most likely candidate
    #     #     probs = torch.softmax(specs, dim=-1)
    #     #     best_candidate = torch.argmax(probs)
            
    #     #     if best_candidate == base:
    #     #         verified_tokens.append(base)
    #     #     else:
    #     #         # Fallback to base model output
    #     #         verified_tokens.append(base)
    #     # return verified_tokens

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    generated_text: str
    processing_time: float

# FastAPI Implementation
app = FastAPI()
model = OptimizedLLM("vicuna-7b-v1.5.Q4_K_M.gguf")

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        start_time = time.time()
        output = model.generate_with_speculative_decoding(
            request.prompt,
            request.max_length,
            request.temperature
        )
        processing_time = time.time() - start_time
        return GenerationResponse(
            generated_text=output,
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)