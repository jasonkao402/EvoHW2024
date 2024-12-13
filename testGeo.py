import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

listDir = [f for f in os.listdir() if f.endswith('.csv')]
df_main = pd.DataFrame(columns=['Generation', 'Best Fitness', 'Mean Fitness', 'Diversity'])
for file in listDir:
    df = pd.read_csv(file)
    df_main = df_main.add(df, fill_value=0)

df_main /= len(listDir)

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(df_main['Generation'], df_main['Best Fitness'], 'r', label='Best Fitness')
plt.plot(df_main['Generation'], df_main['Mean Fitness'], 'b', label='Mean Fitness')
plt.title('Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(df_main['Generation'], df_main['Diversity'], 'g', label='Diversity')
plt.title('Diversity')
plt.xlabel('Generation')
plt.ylabel('Diversity')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

plt.savefig('result.png')
