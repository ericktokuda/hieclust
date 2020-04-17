# Hierarchical clustering


```
mkdir -p /tmp/foo
python src/batch.py --ndims 2 --outdir /tmp/foo/dim2
python src/batch.py --ndims 3 --outdir /tmp/foo/dim3
python src/createfigures.py --pardir /tmp/foo/
python src/analyze.py --pardir /tmp/foo/

```
