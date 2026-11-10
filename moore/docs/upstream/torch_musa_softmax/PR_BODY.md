Title: Work around intermittent fp32 softmax errors on mp_31

On MTT S5000 with muDNN 3.3.0, fp32 softmax can return nonzero probabilities
for masked elements near the end of a row. The added regression fails against
the installed implementation: 12 incorrect values, with a maximum absolute
error of 0.006055878 at `(3988, 514)`, where the CPU reference is zero.

The standalone C++ reproducer links only muDNN and musart and uses synthetic
input. It also reproduces the error:

| Rows × columns | Tail input | Failed calls |
| --- | --- | --- |
| 8128 × 537 | -10000 | 7 / 3000 |
| 8128 × 538 | -10000 | 3 / 3000 |
| 8128 × 539 | 0 | 8 / 3000 |
| 8128 × 540, padded from 537 | -10000 | 0 / 5000 |

This change uses the existing ATen MUSA softmax kernel for last-dimension fp32
rows longer than 512 whose length is not divisible by four. It is gated on
`mp_31` and muDNN 3.3.0 headers. Functional and `out=` calls share the same
dispatch point; backward uses the existing implementation.

The exact defect inside muDNN is still undetermined. This is a workaround for
the observed kernel variant. The version guard is compile-time; changing the
loaded muDNN library without rebuilding does not change it.

### Reproduce

```bash
g++ -O3 -std=c++17 tests/unittest/operator/repro_mudnn_softmax.cpp \
  -I/usr/local/musa/include -L/usr/local/musa/lib \
  -Wl,-rpath,/usr/local/musa/lib -lmudnn -lmusart -o /tmp/repro_softmax
MUSA_VISIBLE_DEVICES=0 /tmp/repro_softmax --columns 537 --repeats 3000
MUSA_VISIBLE_DEVICES=0 /tmp/repro_softmax --columns 537 --pad 4 --repeats 5000
```

Exit 1 indicates incorrect output; exit 2 indicates a setup/API error.

### Validation

Tested on MTT S5000, driver 3.3.5-server, MUSA Toolkit 5.1.0, muDNN 3.3.0,
PyTorch 2.9.1, torch_musa 2.9.1+a18d871.

- Compiled the patched `SoftMax.cpp` in an isolated namespace and registered it
  in a test process. The same failing regression passed all 3000 iterations.
- All 19 added tests passed, covering row-width boundaries, noncontiguous input,
  `out=`, a nondefault stream and autograd.
- Checked all probabilities against CPU for another 3000 calls at 8128 × 537:
  zero failures. Also checked the non-last-dimension muDNN path.
- C++ formatting, Python formatting and `git diff --check` passed.

Based on main at `2838e77d774a8bcc2593126ad07a3314a399d517`. A full build of upstream main, the 5.2.0 host driver and other architectures
have not been tested.

### Follow-up version check

muDNN 3.4.0 with musart 5.2.0 passed the 537/538/539-column cases: 9000 calls,
zero errors observed. With musart 5.1.0 it passed another 3000 calls at 537
columns. The muDNN 3.3.0 control with musart 5.2.0 failed 3 of 3000 calls.
All runs used the same S5000 and unchanged 3.3.5-server host driver.

Closing this PR. The error has only been reproduced with muDNN 3.3.0.
