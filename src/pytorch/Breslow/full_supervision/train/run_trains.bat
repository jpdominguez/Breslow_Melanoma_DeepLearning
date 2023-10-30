FOR /L %%x in (0,1,4) DO (
    python Student_Strongly_Supervised.py -n %%x -b 32 -m densenet121 
)