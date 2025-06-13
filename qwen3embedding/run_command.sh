# you should compile the llama.cpp for cuda in your os, before using llama-cli, or it will only work with CPU.
# export PATH=/opt/llama.cpp/bin:$PATH
# export LD_LIBRARY_PATH="/opt/llama.cpp/bin:$LD_LIBRARY_PATH"

llama-cli -m Qwen3-4B-Q4_K_M.gguf -sys "You are a helpful assistant" -ngl 99 -b 2048 -c 4096 -fa
