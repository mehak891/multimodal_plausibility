import torch
from transformers import LlavaNextForConditionalGeneration, LlavaProcessor,AutoProcessor
import utils 
import math
import torch.nn.functional as F
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class LlavaModel:
    def __init__(self, model_name, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name,use_fast=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)

        self.model.eval()
        self.activations = {}
        self.resid_after_attn_probs = {}
        self.resid_after_mlp_probs = {}
            
        #self.register_hooks()
        self.current_residual = None
        self.handles = []
        self.step = 0

    def register_hooks(self):
        """
        Registers hooks for:
            - gate_proj
            - up_proj
            - down_proj
        For ALL LLaMA layers.

        Returns: activations dict
        """

        self.activations = {}

        def save_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    self.activations[name] = output[1].detach().cpu()
                # Store on CPU to reduce GPU memory load
                else:
                    self.activations[name] = output.detach().cpu()
            return hook

        def save_activation_input(name):
            def hook(module, input, output):
                self.activations[name] = input[0].detach().cpu()
            return hook

        def save_activation_attention(name):
            def hook(module, inputs, output):
                """
                Forward hook for self-attention layer
                """
                # Try to get hidden states
                if len(inputs) > 0:
                    hidden_states = inputs[0]
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                else:
                    # fallback to output if inputs is empty
                    hidden_states = output
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]

                # hidden_states: (B, T, D)
                B, T, D = hidden_states.shape
                if hasattr(module, "num_attention_heads"):
                    H = module.num_attention_heads
                elif hasattr(module, "num_heads"):
                    H = module.num_heads
                else:
                    # fallback
                    H = module.q_proj.out_features // module.head_dim
                head_dim = module.head_dim

                # Q/K projections
                Q = module.q_proj(hidden_states)
                K = module.k_proj(hidden_states)

                Q = Q.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, head_dim)
                K = K.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, head_dim)

                # Compute attention scores
                attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

                # Softmax
                attn_probs = torch.softmax(attn_scores, dim=-1)

                # Keep only last token to save memory
                #if self.save_last_token_only:
                #    attn_probs = attn_probs[:, :, -1, :]  # (B, H, T)

                # Sanitize NaNs (fully masked rows become 0)
                attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
                # Save to CPU immediately
                self.activations[name] = attn_probs.detach().cpu()
                self.activations[f"{name}_output"] = output[0].detach().cpu() if isinstance(output,tuple) else output.detach().cpu()
                del Q, K, attn_scores
            return hook

        def residual_hook(layer_idx):
            def hook(module, inputs, output):
                def unwrap_hidden(x):
                    return x[0] if isinstance(x, tuple) else x
                # inputs[0] = r_l
                # output     = r_{l+1}
                if layer_idx == 0:
                    self.current_residual = None
                    self.step = self.step + 1
                    self.resid_after_attn_probs = {}
                    self.resid_after_mlp_probs = {}
                r_l = inputs[0].detach().cpu()
                r_lplus1 = unwrap_hidden(output).detach().cpu()
                delta_attn = self.activations[f"layer_{layer_idx}_attention_output"]
                delta_mlp = self.activations[f"layer_{layer_idx}_mlp"]
                #print(f"layer {layer_idx} Inputs {inputs[0].shape} Outputs {r_lplus1.shape} Attention {delta_attn.shape} MLP {delta_mlp.shape}")
                # initialize once
                if self.current_residual is None:
                    #print(f"{self.current_residual} is None")
                    self.current_residual = r_l
                
                # after attention residual
                r_after_attn = self.current_residual + self.activations[f"layer_{layer_idx}_attention_output"]
                r_after_mlp = r_after_attn + self.activations[f"layer_{layer_idx}_mlp"]
                self.current_residual = r_lplus1
                # sanity check (optional but recommended)
                # r_lplus1 should equal r_after_attn + mlp_delta
                # we trust HF here

                # update residual stream
                #current_residual = r_lplus1
                seq_len = output[0].shape[1]
                # last token only
                r_attn_last = r_after_attn[:, :, :].to(self.device)
                r_mlp_last  = r_after_mlp[:, :, :].to(self.device)
                #print(f"Attention last {r_attn_last.shape} Logits last {r_mlp_last.shape}")
                final_norm = self.model.language_model.norm

                logits_attn = self.model.lm_head(final_norm(r_attn_last))
                logits_mlp  = self.model.lm_head(final_norm(r_mlp_last))
                self.activations[f"logits_attn_{layer_idx}"] = logits_attn.detach().cpu()
                self.activations[f"logits_mlp_{layer_idx}"] = logits_mlp.detach().cpu()
                #print(f"Logits_attn {logits_attn.shape} Logits mlp {logits_mlp.shape}")
                self.resid_after_attn_probs[layer_idx] = F.log_softmax(logits_attn, dim=-1)
                self.resid_after_attn_probs[layer_idx] = self.resid_after_attn_probs[layer_idx].exp().cpu()

                self.resid_after_mlp_probs[layer_idx]= F.log_softmax(logits_mlp,  dim=-1)
                self.resid_after_mlp_probs[layer_idx] = self.resid_after_mlp_probs[layer_idx].exp().cpu()
                del r_after_attn, r_after_mlp,r_attn_last,r_mlp_last,logits_attn,logits_mlp
            return hook

        clip = self.model.vision_tower.vision_model.embeddings

        self.handles.append(clip.register_forward_hook(save_activation("clip_embeddings")))

        multimodal = self.model.multi_modal_projector

        '''multimodal.linear_1.register_forward_hook(
            save_activation_input(f"multimodal_in")
        )

        multimodal.linear_2.register_forward_hook(
            save_activation(f"multimodal_out")
        )'''
        self.handles.append(multimodal.register_forward_hook(
            save_activation(f"multimodal_projector")
        ))
        llama = self.model.language_model
        self.handles.append(llama.embed_tokens.register_forward_hook(save_activation("llama_embeddings")))

        
        llama_layers = self.model.language_model.layers
        for i, layer in enumerate(llama_layers):
            self.handles.append(layer.self_attn.register_forward_hook(
                save_activation_attention(f"layer_{i}_attention")
            ))
            '''layer.mlp.gate_proj.register_forward_hook(
                save_activation(f"layer_{i}_gate")
            )
            layer.mlp.up_proj.register_forward_hook(
                save_activation(f"layer_{i}_up")
            )
            layer.mlp.down_proj.register_forward_hook(
                save_activation(f"layer_{i}_down")
            )'''
            self.handles.append(layer.mlp.register_forward_hook(
                save_activation(f"layer_{i}_mlp")
            ))
            self.handles.append(layer.register_forward_hook(residual_hook(i)))

    def generate_custom(self, image, question, max_new_tokens, answer, system_prompt="", save_path = None):
        print(f"Inputs to model: {image} {system_prompt} {question}")
        inputs = self.processor(text = f"<image>\n{system_prompt} {question}", images = [image], truncation=False, return_tensors="pt").to(self.device)
        # generation (safe)
        self.register_hooks()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                use_cache=False
            )
        for h in self.handles:
            h.remove()
        self.handles.clear()

        prompt_len = inputs["input_ids"].shape[1]
        answer_tokens = output_ids[0][prompt_len:]
        vision_embeds = self.model.vision_tower(inputs["pixel_values"].squeeze(0)).last_hidden_state
        vision_tokens = self.model.multi_modal_projector(vision_embeds).shape[1]
        gen_ids = answer_tokens
        yes_ids = self.processor.tokenizer("Yes", add_special_tokens=False).input_ids
        gt_ids = self.processor.tokenizer(answer, add_special_tokens=False).input_ids
        no_ids = self.processor.tokenizer("No", add_special_tokens=False).input_ids
        print(f"Vision tokens {vision_tokens}")
        #torch.save(self.activations, "activations.pt")
        #torch.save([self.resid_after_attn_probs,self.resid_after_mlp_probs, gen_ids, gt_ids], "residual.pt")
        #image.save("myimg.png")
        utils.plot_embedding_similarity_and_norms(self.activations["clip_embeddings"],self.activations["multimodal_projector"],self.activations["llama_embeddings"],save_path)
        attn_maps = [self.activations[f"layer_{i}_attention"] for i in range(32)]
        utils.plot_attention_overlay_no_grid(image,attn_maps,vision_tokens,save_path)
        attn_logit = [self.activations[f"logits_attn_{layer_idx}"] for layer_idx in range(32)]
        mlp_logit = [self.activations[f"logits_mlp_{layer_idx}"] for layer_idx in range(32)]
        utils.save_logit_lens_interleaved_gen_vs_gt(self.resid_after_attn_probs,self.resid_after_mlp_probs,gen_ids,gt_ids,yes_ids,no_ids,attn_logit,mlp_logit,save_path)
        return self.processor.decode(
            answer_tokens,
            skip_special_tokens=True
        )
