## Downloading and Installing `bbhx`

### Step 1: Install `cmake` (if not already installed)
Check if `cmake` is installed:
```bash
cmake --version
```
If not installed, run:
```bash
brew install gcc cmake
```

### Step 2: Install LAPACK 3.6.1

`bbhx` requires **LAPACK version 3.6.1**, which is not available via `pip`.

1. Download the tar file from [LAPACK 3.6.1](https://www.netlib.org/lapack/#_lapack_version_3_6_1).
2. Unzip the tar file.
3. Inside the unzipped folder (e.g., `lapack-3.6.1`), create a file named `make.inc` by copying from the template:
   ```bash
   cp make.inc.example make.inc
   ```

### Step 3: Build LAPACK

Navigate to the `lapack-3.6.1` directory and run:
```bash
make
```

#### Possible Errors & Fixes

- **Type mismatch error (`ctrevc3.f`)**  
If you encounter:
```
Error: Type mismatch between actual argument at (1) and actual argument at (2) (REAL(4)/COMPLEX(4)).
```
Edit `ctrevc3.f`, go to line 596, and **replace** `ONE` with `CONE`.

- **Missing `librefblas.a` during testing**
```
ld: file not found: ../../librefblas.a
...
make: *** [blas_testing] Error 2
```
This just means some tests failed — you can ignore this error.

### Step 4: Run Installation
```bash
make install
```
If you see:
```
make: Nothing to be done for 'install'
```
That’s fine — nothing further is required here.

### Step 5: Install `bbhx`
Now install the package:
```bash
pip install bbhx
```
Or, if you want to force a rebuild:
```bash
pip install --no-binary :all: --force-reinstall bbhx
```

### Step 6: Fixing Missing Library Errors

If you get errors like:
- `ld: library 'lapacke' not found`
- `ld: library 'gsl' not found`
- `'lapacke.h' file not found`

Here’s how to fix it:

1. These files do exist, but the linker can’t find them. So you need to export the correct paths.

2. In your terminal:
   ```bash
   # Replace these with your actual absolute paths
   INC_LAPACK_TOP="/Users/akmdgreat/Desktop/LISA/lisa_code/bbhx/lapack-3.6.1"
   INC_CONDA="$CONDA_PREFIX/include"
   INC_LAPACKE="/Users/akmdgreat/Desktop/LISA/lisa_code/bbhx/lapack-3.6.1/LAPACKE/include"

   export CPPFLAGS="-I${INC_LAPACK_TOP} -I${INC_CONDA} -I${INC_LAPACKE} ${CPPFLAGS:-}"
   export LDFLAGS="-L${LAPACKE_LIB} -Wl,-rpath,${LAPACKE_LIB} \
                   -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib ${LDFLAGS}"
   ```

3. Verify the path settings:
   ```bash
   echo $CPPFLAGS
   ```
   Expected output:
   ```
   -I/Users/akmdgreat/Desktop/LISA/lisa_code/bbhx/lapack-3.6.1 
   -I/Users/akmdgreat/anaconda3/envs/bethenv/include 
   -I/Users/akmdgreat/Desktop/LISA/lisa_code/bbhx/lapack-3.6.1/LAPACKE/include
   ```
   (Duplicate paths are okay.)
