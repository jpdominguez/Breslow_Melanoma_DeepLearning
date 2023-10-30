FOR /L %%x in (10,1,14) DO (
    python Student_train_pseudo_and_strong.py -n %%x -b 32 -e 100
)