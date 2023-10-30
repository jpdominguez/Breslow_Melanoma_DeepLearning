FOR /L %%x in (10,1,19) DO (
    python Student_Strongly_Supervised.py -n %%x -b 32 -e 100
)