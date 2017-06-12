### 第一章 用Python开始机器学习之旅

机器学习，顾名思义，指导机器去学着独自完成自己的任务，就这么简单。复杂的是其中的具体实现细节，这也正是你读这本书的原因。

也许你有特别多的数据，却没有足够的鉴别能力。你希望使用机器学习算法让你能够完成这种挑战。所以你尝试开始研究各类算法。但是过了一段时间以后你会困惑：如此大量的算法，到底哪个是我应该选择的呢？

或者说，大体上你是对机器学习感兴趣的，同时你还常常研读机器学习相关的博客和文章。一切都看起来如此酷炫、奇妙。于是你开始了自己的探索，比如给决策树算法或者支持向量机（SVM）填充一些测试数据。然后在你成功地应用这些算法到其他数据后，你会疑惑：整个设置确实正确么？我确实得到了最理想的结果么？我怎么知道是否存在更好的算法呢？或者我的数据确实正确么？

欢迎来到我们的俱乐部！我们大家（包括作者） 都处于这样一种阶段：寻找着一些用来结石隐藏在机器学习理论书籍背后知识的信息。确实这些信息被成为“黑色艺术”，通常在标准的教科书中是学不到的。总而言之，我们为了你们这些年轻人写这本书，一本不仅能让你们快速入门，同时也能带着你们边学边练的书。我们衷心希望它能够帮助你们很顺利地进入这个计算机科学的最激动人心的领域之一。

## 机器学习和Python - 一个梦幻组合

机器学习的目标是教机器（软件）通过提供给它们一大堆样例（如何去做任务或者不去做任务）去完成这个任务。让我们假设一下：每天早上当你打开你的电脑，你总是做相同的事情：归档Email（让相同主题的文件放在同一个文件夹）。过了一段时间以后，你会感觉到厌烦，而去思考如何自动化这些该死的事情。有一种方式：在你归档邮件的时候，开始分析你的大脑，写下你的大脑处理过程中的所有规则。然而，这将是相当繁琐的，且总不太完善。有时候你会遗漏一些规则，而又有时候你会重复设定规则。一个更好的，更为先进的证明方法是：通过选择一组Email的元数据和邮件内容/文件夹名称 这种 字符对，且用算法来生成最好的规则集合  来自动化这种过程。这些字符对将成为你的测试数据，结果规则集合（或者叫模型）将会被应用于以后那些我们尚未收到的邮件中。 这个过程就是最简单的机器学习。

当然，机器学习（通常也称为数据挖掘或预测分析）本身并不是一个全新的领域。恰恰相反，它近年来的成功可以归功于：应用坚实技术的务实方法和其他成功领域的见解，如统计学。我们的目的是让人类对数据有深入的了解，例如，通过学习 更多关于基本模式和关系。当你阅读更多关于成功应用机器学习（你已经看过www.kaggle.com，不是吗？），你会看到，应用统计学是一种常见的机器学习领域的手段。

正如您稍后将看到的，实现一个像样机器学习方法的过程决不是一个瀑布式的过程。相反，你会发现自己在分析过程中，来回不断地尝试各种不同版本的输入数据，应用于各种不同的算法中。正是这种机器学习这种探索性的天性完全适合用Python实现。作为一种解释的高级编程语言，Python正是为了这种尝试不同的事情的任务而设计的。更重要的是，它能以更快地速度做到这一点。当然，它比C或类似的静态类型编程语言慢。不管怎么说，与易于使用的库（通常是用C写的），你不必牺牲速度。

## 这本书将教会你什么（什么也没有）

这本书将让你大致了解什么类型的学习算法是目前最常用于不同领域的机器学习算法，以及在应用时要注意哪些。从我们自己的经验来看，我们知道，做“酷”的东西，也就是说，使用和调整机器学习算法，如支持向量机，最近邻搜索，或合奏，将 只是消费的一个很好的机器学习专家的总时间的一小部分。看下面的典型的工作流程，我们看到大部分的时间会花在相当平凡的任务：





This book will give you a broad overview of what types of learning algorithms are currently most used in the diverse fields of machine learning, and where to watch out when applying them. From our own experience, however, we know that doing the “cool” stuff, that is, using and tweaking machine learning algorithms such as support vector machines, nearest neighbor search, or ensembles thereof, will only consume a tiny fraction of the overall time of a good machine learning expert. Looking at the following typical workflow, we see that most of the time will be spent in rather mundane tasks: 

这本书将让你大致了解什么类型的学习算法是目前最常用的不同领域的机器学习，以及在哪里注意应用时。从我们自己的 然而，经验，我们知道，做“酷”的东西，也就是说，使用和调整机器学习算法，如支持向量机，最近邻搜索，或合奏，将 只是消费的一个很好的机器学习专家的总时间的一小部分。看下面的典型的工作流程，我们看到大部分的时间会花在相当平凡的任务：



Reading in the data and cleaning it 

读取数据并进行清理



Exploring and understanding the input data 

对输入数据的探索和理解



Analyzing how best to present the data to the learning algorithm 

分析如何最好地将数据呈现给学习算法



Choosing the right model and learning algorithm 

选择合适的模型和学习算法



Measuring the performance correctly 

正确测量性能



When talking about exploring and understanding the input data, we will need a bit of statistics and basic math. However, while doing that, you will see that those topics that seemed to be so dry in your math class can actually be really exciting when you use them to look at interesting data. 

当谈到探索和理解的输入数据，我们需要一点统计和数学基础。然而，这么做的时候，你会看到那些话题，似乎在你这么干 当你使用数学课看有趣的数据时，你的数学课确实很令人兴奋。



The journey starts when you read in the data. When you have to answer questions such as how to handle invalid or missing values, you will see that this is more an art than a precise science. And a very rewarding one, as doing this part right will open your data to more machine learning algorithms and thus increase the likelihood of success. 

当你的旅程开始读取数据。当你要回答诸如如何处理无效或缺失值的问题，你会发现这是一门艺术，不是一门精确的科学。和一个版本的 当你做这个部分的时候，你会把你的数据打开到更多的机器学习算法中，从而增加成功的可能性。



With the data being ready in your program’s data structures, you will want to get a real feeling of what animal you are working with. Do you have enough data to answer your questions? If not, you might want to think about additional ways to get more of it. Do you even have too much data? Then you probably want to think about how best to extract a sample of it. 

随着数据在你的程序数据结构中被准备好，你会想要真正了解你正在使用的动物。你有足够的数据来回答你的问题吗？如果没有，你会 我不想考虑更多的方法来获取更多信息。你是否有太多的数据？然后，您可能想考虑如何最好地提取它的样本。



Often you will not feed the data directly into your machine learning algorithm. Instead you will find that you can refine parts of the data before training. Many times the machine learning algorithm will reward you with increased performance. You will even find that a simple algorithm with refined data generally outperforms a very sophisticated algorithm with raw data. This part of the machine learning workflow is called feature engineering, and is most of the time a very exciting and rewarding challenge. You will immediately see the results of being creative and intelligent. 

你通常不会把数据直接进入你的机器学习算法。相反，你会发现你可以细化培训前的数据部分。机器学习算法有多少次 我会奖励你增加的表现。您甚至会发现，一个简单的算法，细化数据通常优于一个非常复杂的算法与原始数据。机器的这部分 关于工作流为特征的工程，而大部分时间是一个非常令人兴奋的和有意义的挑战。您将立即看到创新和智能化的结果。



Choosing the right learning algorithm, then, is not simply a shootout of the three or four that are in your toolbox \(there will be more you will see\). It is more a thoughtful process of weighing different performance and functional requirements. Do you need a fast result and are willing to sacrifice quality? Or would you rather spend more time to get the best possible result? Do you have a clear idea of the future data or should you be a bit more 

选择正确的学习算法，然后，不只是一个枪战的三或四，在你的工具箱（将有更多你会看到）。它更是一个体贴的衡量不同的过程 租金的性能和功能要求。你需要一个快速的结果，愿意牺牲质量？或者你更愿意花更多的时间去获得最好的结果？你有一个清晰的我 未来数据的DEA还是应该多一点？



conservative on that side? 

保守吗？



Finally, measuring the performance is the part where most mistakes are waiting for the aspiring machine learner. There are easy ones, such as testing your approach with the same data on which you have trained. But there are more difficult ones, when you have imbalanced training data. Again, data is the part that determines whether your undertaking will fail or succeed. 

最后，测量性能是大部分错误等待有抱负的机器学习者的一部分。有一些简单的方法，比如用相同的数据测试你的方法。 训练.但是还有更难的，当你有不平衡的训练数据。再次，数据的一部分，决定了你的事业会成功或失败。



We see that only the fourth point is dealing with the fancy algorithms. Nevertheless, we hope that this book will convince you that the other four tasks are not simply chores, but can be equally exciting. Our hope is that by the end of the book, you will have truly fallen in love with data instead of learning algorithms. 

我们看到，只有第四点是处理花哨的算法。然而，我们希望这本书能让你相信，其他四个任务不是简单的家务事，但同样可以激励 惯性导航与制导.我们希望在书的结尾，你会真正爱上数据而不是学习算法。



To that end, we will not overwhelm you with the theoretical aspects of the diverse ML algorithms, as there are already excellent books in that area \(you will find pointers in the Appendix\). Instead, we will try to provide an intuition of the underlying approaches in the individual chapters—just enough for you to get the idea and be able to undertake your first steps. Hence, this book is by no means the definitive guide to machine learning. It is more of a starter kit. We hope that it ignites your curiosity enough to keep you eager in trying to learn more and more about this interesting field. 

为此，我们不会用不同ML算法的理论方面来压倒你，因为在这方面已经有了优秀的书籍（你会在附录中找到指针）。相反，W E将尝试提供一个直觉的基本方法，在个别章节，足以让你得到的想法，并能够进行你的第一步。因此，这本书并非毫无意义。 机器学习的权威指南。它更像是一个初学者工具包。我们希望它能激发你的好奇心，使你渴望学习更多关于这个有趣领域的知识。



In the rest of this chapter, we will set up and get to know the basic Python libraries NumPy and SciPy and then train our first machine learning using scikit-learn. During that endeavor, we will introduce basic ML concepts that will be used throughout the book. The rest of the chapters will then go into more detail through the five steps described earlier, highlighting different aspects of machine learning in Python using diverse application scenarios. 

在本章的其余部分，我们将了解基本的Python库numpy和scipy然后训练我们的第一个机器学习使用scikit学习。在这一努力中，我们将介绍 减少基本毫升概念将贯穿全书。本章的其余部分将进入更详细的通过五个步骤之前描述的，突出了机器的不同方面 使用不同的应用场景在Python中学习。







