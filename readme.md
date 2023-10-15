# Case 3 (Computer Vision)

1. Download data archive from kaggle to `data` folder
2. Extract archive `unzip data/case3-datasaur-photo.zip` 
3. Run `docker build . -t case3`
4. Run `docker run -d -v .:/workdir case3 python main.py`
5. *data/best_model.pth* contains model; *output.csv* contains answers;


[link to presentation](https://docs.google.com/presentation/d/1OAsUxfSiq21omr0SxV8V-Ohv6FYOOD9s3NZ3YPj9Sqc/edit?usp=sharing)

P.S. *main.py* and *notebook.ipynb* are the same.