import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(17, 6))

listDir = [f for f in os.listdir() if f.startswith('Run_')]
for i, dir in enumerate(listDir):
    print(f'{i:02d}: {dir}')
idx = int(input('Select a directory: '))
listDir2 = [f for f in os.listdir(listDir[idx]) if f.endswith('.csv')]
listDir2.sort()

df_avg = pd.DataFrame(columns=['Generation', 'Best Fitness', 'Mean Fitness', 'Distance'])
for i in range(len(listDir2)):
    df = pd.read_csv(f'{listDir[idx]}/{listDir2[i]}')
    df_avg = df_avg.add(df, fill_value=0)
    plt.subplot(1, 2, 1)
    plt.plot(df['Generation'], df['Best Fitness'], 'r', alpha=0.2)
    plt.plot(df['Generation'], df['Mean Fitness'], 'b', alpha=0.2)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['Generation'], df['Distance'], 'g', alpha=0.2)

plt.subplot(1, 2, 1)
plt.title('Fitness')
plt.plot(df['Generation'], df_avg['Best Fitness'] / len(listDir2), 'r', label='Best Fitness')
plt.plot(df['Generation'], df_avg['Mean Fitness'] / len(listDir2), 'b', label='Mean Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.title('Distance')
plt.plot(df['Generation'], df_avg['Distance'] / len(listDir2), 'g', label='Distance')
plt.xlabel('Generation')
plt.ylabel('Distance')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f'result_{listDir[idx]}.png')
plt.show()
