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
A: