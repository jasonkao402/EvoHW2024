# hw2
Q: In many ways, using the word "creativity" in a field of computer science opens up a Pandora's box of controversy. Convincing people that our creativity can be enhanced using computer software is something difficult. Arguing with people that the techniques themselves might be capable of creativity is often futile. But there are an increasing number of researchers prepared to try.

The quote above is from a 2002 article "An Introduction to Creative Evolutionary Systems" by Peter J. Bentley and David W. Corne. Has anything changed today, either in terms of technology or how people think about the subject? Write down your opinion and give actual examples to support your argument.

A: In late 2022, Stable Diffusion was introdced, which is a novel technique that "creates" art by removing successive applications of Gaussian noise on training images, which can be thought of as a sequence of denoising autoencoders. People started to generate art using the official model and community models, however, the training dataset for the community models quickly became a controversial issue. The training dataset was mostly collected from the internet, and it was not clear whether the images were used with permission. The process of "creating" art using algorithms is also challenged, whether the art is truly original or just a mix and match of existing art.

Tool assisted creativity is a topic that has been discussed for a long time, this has happened before when digital cameras were introduced. I think creativity still lies in the hands of the creator, give the same tool to a group of people that is familiar with art and those don't, the result will be drastically different. The tool can help the creator to express their creativity, but it is not the creativity itself. The same goes for the Stable Diffusion, the model is just a tool to generate art, the creativity comes from the people who use the model. In conclusion, wheather it is a canvas, a digital camera, or a diffusion model, the tool is just a medium to express the creativity. 

---

Q: from Terence Tao's lecture on mathematics as a creative subject

So is mathematics a creative subject? Definitely.

When you see mathematics in a school context, it's often presented in a somewhat dry manner that there's certain recipes that you have to follow in order to solve a problem. And if you deviate, then you get your marks deducted or something. But when you're a research mathematician, you are solving problems that standard techniques don't quite apply. Because it's so abstract and not necessarily tethered to reality, it allows you to be very creative and very flexible.

In the real world, you may have a problem where you have to say, some finite number of resources. You only have X amount of dollars to thwart a problem. You may only have so much time. But in mathematics, you can change the parameters. You may say, OK, what have I had a billion dollars? Could I solve this problem? Or what if I had an infinite amount of manpower? It gives you a lot of flexibility to change the problem into one that maybe you can solve first and then you can from there, go back to solve the actual problem. And that's a freedom that you just don't have in the real world. You can't just say, oh, before I solve this problem, can I first have a billion dollars to experiment.

Yeah, the abstraction that mathematics has afforded a lot of creative freedom.
Regarding the creative freedom mentioned in Terence Tao's talk, what do you think of evolutionary computation inspired by natural evolution? Describe the creative freedom you can have in evolutionary algorithms, especially those that cannot be achieved in the real-world natural evolutionary process.

A: I think the most significant difference between natural evolution and evolutionary computation is the ability to simulate a large number of generations in a short amount of time, but this comes with a drawback that the simulation is not perfect and oversimplified. In evolutionary computation, we often have the freedom to limit the search space, change the mutation rate, and even change the fitness function to our liking. This allows us to explore the search space more efficiently and find solutions that are not possible in the real world. For example, we can change the mutation rate to a very high value to explore the search space more aggressively, or we can change the fitness function to favor certain solutions. In the real world, the mutation rate cannot exceed a certain threshold before the genetic material becomes unstable, and the fitness function is determined by the environment, which is not under our control.  

Q: What was your choice of genotype for encoding the chessboard configuration?
A: Permutation

Q:Briefly describe your design of the genetic operators (mutation and crossover) respectively for the genotype your choose.
A:Swap Mutation: This mutation operator randomly selects two positions in the permutation and swaps their values.
Order Crossover: This crossover operator randomly selects a subset of the parent chromosomes and copies them to the offspring, then fills in the remaining positions with the remaining values from the other parent in the order they appear.

Q:In engineering problems, conditions are often given, and our creativity is mainly focused on performance improvement. During this semester, you've acquired a considerable amount of evolutionary computing knowledge or methods. if we extend the eight-queen problem to n-queens, how do you improve the genotype/representation or operator design? What do you want to achieve with the changes you make?
A: From what I have learned, reducing the search space and using heuristics to guide the search are two common strategies to improve the quality of the solution. For the n-queen problem, I think the permutation representation is still a good choice, improving the mutation and crossover operators to explore the search space more efficiently is the key to solving the problem. For example, we can use a adaptive mutation rate that decreases over time to explore the search space more aggressively in the beginning and converge to a local optimum in the end. We can also use a heuristic crossover operator that favors the safer areas that are less likely to conflict with other queens.  

Q: If we tweak the problem itself a bit: out of n queens, three are particularly close and can see each other, while all the other queens still can't, does your choice of genotype and operators still apply? Why or why not?
A: The three queens are essentially grouped together and need be treated as a separate constraint in the problem.
Let's focus on the three queens that can see each other, I think the permutation representation is still a good choice for this problem,
slight change to encode the three queens as a group will do the trick, but the mutation and crossover operators need to be heavily modified to take into account the special relationship between the three queens. For example, we can add a constraint to the mutation operator that prevents the three queens from being swapped with each other. We can also modify the crossover operator to favor the positions that are less likely to conflict with the three queens. 

Q: How do you design the fitness function for the tweaked version of n-queen problem (with a trio)?
A: The fitness function should take into account the number of conflicts between the queens and the distance between the three queens. The fitness function should penalize the solutions that have conflicts between the queens and reward the solutions that have the three queens inside each other's attack range.

Q: Please conduct a test of your design concerning 'n-queen with a trio' and provide a brief description of your experiment and its results. Share any insights or lessons you have acquired from it.
A: I implemented both the original n-queen problem and the n-queen with a trio problem using the permutation representation, population size of 64, 400 generations, other parameters are changed with intuition, so I won't go into details. Adaptive mutation rate is easy to implement so I used it for both problems, but I kept the crossover operators the same as the original n-queen problem, as I think the crossover operator is too complex to modify for the n-queen with a trio problem.
From my test results, its really hard solve the last 1 or 2 queens in the original n-queen problem when n is larger than 10, however, the modified n-queen came to my surprise as the imperfections in the last 1 or 2 queens can be easily solved as we don't have to worry about the three queens that can see each other.