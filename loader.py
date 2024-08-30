import json
import math
import os
import traceback
import argparse

import chromadb
import torch.multiprocessing as mp
from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models
from marker.output import markdown_exists, save_markdown
from marker.pdf.extract_text import get_length_of_text
from marker.pdf.utils import find_filetype
from marker.settings import settings
from tqdm import tqdm
from graphrag.index.cli import index_cli

os.environ["IN_STREAMLIT"] = "true"  # Avoid multiprocessing inside surya
os.environ["PDFTEXT_CPU_WORKERS"] = "1"  # Avoid multiprocessing inside pdftext
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS

configure_logging()


def worker_init(shared_model):
    if shared_model is None:
        shared_model = load_all_models()

    global model_refs
    model_refs = shared_model


def worker_exit():
    global model_refs
    del model_refs


def process_single_pdf(args):
    filepath, out_folder, metadata, min_length = args

    fname = os.path.basename(filepath)
    if markdown_exists(out_folder, fname):
        return

    try:
        # Skip trying to convert files that don't have a lot of embedded text
        # This can indicate that they were scanned, and not OCRed properly
        # Usually these files are not recent/high-quality
        if min_length:
            filetype = find_filetype(filepath)
            if filetype == "other":
                return 0

            length = get_length_of_text(filepath)
            if length < min_length:
                return

        full_text, images, out_metadata = convert_single_pdf(filepath, model_refs, metadata=metadata, batch_multiplier=1)
        if len(full_text.strip()) > 0:
            save_markdown(out_folder, fname, full_text, images, out_metadata)
        else:
            print(f"Empty file: {filepath}.  Could not convert.")
    except Exception as e:
        print(f"Error converting {filepath}: {e}")
        print(traceback.format_exc())


def multiple(in_folder, out_folder):
    chunk_idx = 0
    num_chunks = 1
    max = None
    workers = 10
    meta = None
    min_len = None

    files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
    files = [f for f in files if os.path.isfile(f)]
    os.makedirs(out_folder, exist_ok=True)

    # Handle chunks if we're processing in parallel
    # Ensure we get all files into a chunk
    chunk_size = math.ceil(len(files) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size
    files_to_convert = files[start_idx:end_idx]

    # Limit files converted if needed
    if max:
        files_to_convert = files_to_convert[:max]

    metadata = {}
    if meta:
        metadata_file = os.path.abspath(meta)
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    total_processes = min(len(files_to_convert), workers)

    # Dynamically set GPU allocation per task based on GPU ram
    if settings.CUDA:
        tasks_per_gpu = settings.INFERENCE_RAM // settings.VRAM_PER_TASK if settings.CUDA else 0
        total_processes = int(min(tasks_per_gpu, total_processes))
    else:
        total_processes = int(total_processes)

    try:
        mp.set_start_method('spawn')  # Required for CUDA, forkserver doesn't work
    except RuntimeError:
        raise RuntimeError("Set start method to spawn twice. This may be a temporary issue with the script. Please try running it again.")

    if settings.TORCH_DEVICE == "mps" or settings.TORCH_DEVICE_MODEL == "mps":
        print(
            "Cannot use MPS with torch multiprocessing share_memory. This will make things less memory efficient. If you want to share memory, you have to use CUDA or CPU.  Set the TORCH_DEVICE environment variable to change the device.")

        model_lst = None
    else:
        model_lst = load_all_models()

        for model in model_lst:
            if model is None:
                continue
            model.share_memory()

    print(
        f"Converting {len(files_to_convert)} pdfs in chunk {chunk_idx + 1}/{num_chunks} with {total_processes} processes, and storing in {out_folder}")
    task_args = [(f, out_folder, metadata.get(os.path.basename(f)), min_len) for f in files_to_convert]

    with mp.Pool(processes=total_processes, initializer=worker_init, initargs=(model_lst,)) as pool:
        list(tqdm(pool.imap(process_single_pdf, task_args), total=len(task_args), desc="Processing PDFs", unit="pdf"))

        pool._worker_handler.terminate = worker_exit

    # Delete all CUDA tensors
    del model_lst


def single(fname, output):
    model_lst = load_all_models()
    full_text, images, out_meta = convert_single_pdf(fname, model_lst, max_pages=None, langs=None, batch_multiplier=1, start_page=None)

    fname = os.path.basename(fname)

    subfolder_path = save_markdown(output, fname, full_text, images, out_meta)

    print(f"Saved markdown to the {subfolder_path} folder")


def chunk(input_):
    chunks = []
    ids = []
    id_ = 0

    for file in os.listdir(input_):

        print(file, os.listdir(input_))
        with open(os.path.join(input_, file), 'r') as file_:
            file_contents = file_.readlines()

        for line in file_contents:
            if len(line.split(' ')) < 5:
                continue
            else:
                chunks.append(line)
                ids.append('id' + str(id_))
                id_ += 1

    return chunks, ids


def load_text_chroma(input_, output_, collection_name, create_collection=False):
    client = chromadb.PersistentClient(output_, settings=chromadb.Settings(anonymized_telemetry=False))

    if create_collection:
        collection = client.create_collection(collection_name)
    else:
        collection = client.get_collection(collection_name)

    documents, ids = chunk(input_)

    collection.add(
        documents=documents,
        ids=ids
    )


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--convert', help='Converts PDFs to markdown (Make sure to have a pdf subfolder in your input folder)',
                    action='store_true')
parser.add_argument('-n', '--newcollection', help='Create a new collection, and restart the GraphRAG index. (Only do this if this is your first '
                                                   'time running this.)', action='store_true')
parser.add_argument('--input_folder', help='The input folder to read files from. (Defaults to `./input`)', default='./input')
parser.add_argument('--graphrag_input', help='The root folder for the graphRAG project (The folder which contains your settings.yaml)', default='.')
parser.add_argument('--chroma_dir', help='The location of your chromaDB database', default='./.chroma')
parser.add_argument('--collection_name', help='The name of your chromaDB collection', default='collection1')
args = parser.parse_args()

if args.convert:
    print('CONVERTING PDFs...')
    multiple(os.path.join(args.input_folder, 'pdfs'), os.path.join(args.input_folder, 'markdown'))

print('CREATING CHROMADB...')
load_text_chroma(os.path.join(args.input_folder, 'markdown'), args.chroma_dir, args.collection_name, args.newcollection)

print('CREATING GRAPHRAG INDEX...')
index_cli(args.graphrag_input, False, False, None, False, False, None, None, None, False)
