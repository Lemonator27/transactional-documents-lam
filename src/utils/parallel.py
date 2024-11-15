import os
from multiprocessing import Pool
from typing import Callable, List, Optional, TypeVar

from tqdm import tqdm

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Return type

def pmap(
    func: Callable[[T], R],
    items: List[T],
    num_processes: Optional[int] = None,
    desc: Optional[str] = None,
    use_tqdm: bool = True,
    chunksize: Optional[int] = 1
) -> List[R]:
    """
    Generic parallel processing function with progress bar.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        num_processes: Number of processes to use (defaults to CPU count)
        desc: Description for the progress bar
        use_tqdm: Whether to show progress bar
        chunksize: Size of chunks for multiprocessing (defaults to 1)
    
    Returns:
        List of results in the same order as input items
    """
    if not items:
        return []
    
    if num_processes is None:
        if 'NUM_CPUS' in os.environ:
            num_processes = int(os.environ['NUM_CPUS'])
        else:
            num_processes = os.cpu_count() // 2
    
    if chunksize is None:
        chunksize = max(1, len(items) // num_processes)
        
    with Pool(processes=num_processes) as pool:
        if use_tqdm:
            results = list(tqdm(
                pool.imap(func, items, chunksize=chunksize),
                total=len(items),
                desc=desc
            ))
        else:
            results = pool.map(func, items, chunksize=chunksize)
            
    return results

if __name__ == '__main__':
    def square(x):
        return x * x  
  
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    num_workers = 2
    
    squared_data = pmap(square, data, num_processes=num_workers)  
    print(squared_data)  # Output: [1, 4, 9, 16, 25]  
