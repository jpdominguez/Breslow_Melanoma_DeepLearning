FOR /L %%x in (0,1,9) DO (
    python Student_train_pseudo_and_strong.py -n %%x -b 32 -e 100
)