import torch
import time
import sys
import threading

def gpu_worker(gpu_id, stop_event):
    """
    Worker function to keep a single GPU busy with continuous computation.
    """
    device = torch.device(f"cuda:{gpu_id}")
    
    try:
        # Try larger tensor first
        tensor_size = 90000
        a = torch.randn(tensor_size, tensor_size, device=device)
        b = torch.randn(tensor_size, tensor_size, device=device)
        print(f"GPU {gpu_id}: Allocated {tensor_size}x{tensor_size} tensors")
    except torch.cuda.OutOfMemoryError:
        # Fall back to smaller tensors
        tensor_size = 40000
        try:
            a = torch.randn(tensor_size, tensor_size, device=device)
            b = torch.randn(tensor_size, tensor_size, device=device)
            print(f"GPU {gpu_id}: Allocated {tensor_size}x{tensor_size} tensors (fallback)")
        except torch.cuda.OutOfMemoryError:
            print(f"GPU {gpu_id}: Cannot allocate even small tensors", file=sys.stderr)
            return
    
    print(f"GPU {gpu_id}: Starting continuous computation...")
    
    while not stop_event.is_set():
        try:
            # Perform multiple operations to keep GPU busy
            c = torch.matmul(a, b)
            time.sleep(30)  # Short sleep to prevent 100% CPU usage
            # d = torch.matmul(b, a)
            # e = c + d
            # f = torch.relu(e)
            # g = torch.softmax(f, dim=1)
            # # Update tensors to create continuous work
            # a = g[:tensor_size, :tensor_size]
            # b = torch.transpose(a, 0, 1)
        except Exception as e:
            print(f"GPU {gpu_id}: Error during computation: {e}", file=sys.stderr)
            break

def main():
    """
    Main function to occupy all visible GPUs with high utilization.
    """
    while True:
        # Check for CUDA devices with zero memory usage
        num_gpus = torch.cuda.device_count()
        available_gpus = []
        if num_gpus != 0:
            for i in range(num_gpus):
                mem_info = torch.cuda.memory_stats(i)
                used_mem = mem_info.get('allocated_bytes.all.current', 0)
                if used_mem > 500:
                    print(f"GPU {i}: {used_mem / (1024**3):.2f} GB")
                else:
                    available_gpus.append(i)
            if len(available_gpus) > 0:
                break
            else:
                print("Waiting for all GPUs to be free...")
        else:
            print("No visible CUDA devices found.")

        time.sleep(30)

    print(f"Found {available_gpus} visible CUDA device(s).")
    
    # Event to signal threads to stop
    stop_event = threading.Event()
    threads = []
    
    # Start a worker thread for each GPU
    for i in available_gpus:
        thread = threading.Thread(target=gpu_worker, args=(i, stop_event))
        thread.start()
        threads.append(thread)
    
    print("All GPUs are now running continuous computations. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping all GPU workers...")
        stop_event.set()
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        
        print("All GPU workers stopped. Exiting.")

if __name__ == "__main__":
    main()
