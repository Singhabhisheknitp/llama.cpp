ldd ../build/bin/llama-simple >> requirement.txt

cp ../build/bin/llama-simple .
cp ../build/src/libllama.so .
cp ../build/ggml/src/libggml.so .
cp ../build/ggml/src/libggml-base.so .
cp ../build/ggml/src/libggml-cpu.so .

cp ../huggingface/tiny-llama-stories-42m/tiny-lllama-stories-42m.gguf .