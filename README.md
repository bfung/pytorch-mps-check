Checks to see the Metal / GPU compatibility for pytorch.

Used this while figuring out if stable diffusion could run faster on my laptop:
```
macOS Catalina 10.15.7
MacBook Pro (Retina, 15-inch, Early 2013)  <-- yes, almost 10 year old computer
Processor 2.7 GHz Quad-Core Intel Core i7
Memory 16 GB 1600 MHz DDR3
Graphics Intel HD Graphics 4000 1536 MB    <-- no GPU, old graphics card
```
## Notes

* CUDA support dropped from macOS after 10.13 (High Sierra).
* Metal support in pytorch on supports macOS 12.3+ (Monterey) and later.
  * This leaves: 

      macOS Big Sur	  11.6.8
      macOS Catalina  10.15.7
      macOS Mojave	  10.14.6

    in the cold, no `torch.backend.mps`.

## Usage

1. Install dependencies (use a virtualenv for best practices)
2. run the script

Copypasta

    $ python3 -m venv venv
    $ . venv/bin/activate
    (venv) $ pip install -r requirements.txt
    (venv) $ ./detect_gpu_features.py