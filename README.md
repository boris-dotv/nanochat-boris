# nanochat

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## Talk to it

To get a sense of the endpoint of this repo, you can currently find [nanochat d32](https://github.com/karpathy/nanochat/discussions/8) hosted on [nanochat.karpathy.ai](https://nanochat.karpathy.ai/). "d32" means that this model has 32 layers in the Transformer neural network. This model has 1.9 billion parameters, it was trained on 38 billion tokens by simply running the single script [run1000.sh](run1000.sh), and the total cost of training was ~$800 (about 33 hours training time on 8XH100 GPU node). While today this is enough to outperform GPT-2 of 2019, it falls dramatically short of moden Large Language Models like GPT-5. When talking to these micro models, you'll see that they make a lot of mistakes, they are a little bit naive and silly and they hallucinate a ton, a bit like children. It's kind of amusing. But what makes nanochat unique is that it is fully yours - fully configurable, tweakable, hackable, and trained by you from start to end. To train and talk to your own, we turn to...

## Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensates by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.

## Running on CPU / MPS

nanochat cn be run on CPU or on MPS (if you're on Macbook), and will automatically try to detect what device is best to run on. You're not going to get too far without GPUs, but at least you'll be able to run the code paths and maybe train a tiny LLM with some patience. For an example of how to make all the run commands much smaller (feel free to tune!), you can refer to [dev/runcpu.sh](dev/runcpu.sh) file. You'll see that I'm essentially restricting all scripts to train smaller models, to run for shorter number of iterations, etc. This functionality is new, slightly gnarly (touched a lot of code), and was merged in this [CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88) on Oct 21, 2025.

## Customization

To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into midtraining and SFT stages.

## Questions

nanochat is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

This includes all py, rs, html, toml, sh files, excludes the `rustbpe/target` folder, and chooses the cxml output format. Everything is written to the `packaged.txt` file, which atm measures ~330KB (i.e. well below ~100K tokens for a state of the art LLM), and ~8K lines of code in 45 files.

Alternatively, I recommend using [DeepWiki](https://deepwiki.com/) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.

## Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## Contributing

nanochat is nowhere finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

I am looking for someone to be the "nanochat repo czar" to help me manage the nanochat repo and its issues and PRs and be the first round of defense. Examples of work include merging simple fixes (docs, typos, clear and simple bugs etc.), rejecting vibe coded PRs, managing the Issues/PRs, doing brief "sanity check testing" of PRs on the two officially supported platforms (Linux/GPU and Macbook), organizing information into brief updates and highlights for me. We'd be in touch on DMs on Discord or X or whatever is easiest. For your services to the repo you will be listed and linked to under acknowledgements as the nanochat repo czar. Position is at-will so you can contribute for a while and then "resign" at any time later, totally ok and thank you for your help, just me know. Apply via DM to me on X, thank you!

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT


## 20251023 git 问题记录


起初, 我 git clone 了 Karpathy 的 nanochat, 称当时最新版本是 V5, 与此同时我设置了 git remote add upstream https://github.com/karpathy/nanochat.git, 此时有:

```
Karpathy 的 nanochat: V1 -> V2 -> V3 -> V4 -> V5 [master]
你的 nanochat: V1 -> V2 -> V3 -> V4 -> V5 [master]
```


后来一段时间内, 我添加了 RoPE 的注释, 得到了 V5-注释版, 而 Karpathy 和他的合作者们修复了一些 bug, 得到 V6, 于是分岔出现了:

```
                              +--> V6 [Karpathy 的 master]
                             /
V1 -> ... -> V5 (共同的祖先)
                             \
                              +--> V5-注释版 [你的 master]   
```

随后我 set-url 到 boris-dotv/nanochat-boris, 告诉 git, 以后把我的更新都推送到这个 repository 里面, 

git fetch upstream 只是把 V6 下载到本地, 存储在一个 upstream/master 的远程跟踪分支里面, 没有对 V5-注释版做任何操作. 此时若运行 git status, 我们会得到一些终端返回信息:
1. 当前在 master 分支;
2. 当前分支和 upstream/master 分支有分岔;
3. 两个分支都基于同一个版本各自前进了一步.

当前目的: 将 V6 和 V5-注释版整合到一起.

* 方法1: Merge
git merge upstream/master 会创造一个全新的 V7-合并版, 其有两个父提交: V6 和 V5-注释版:
```
             +-----> V6 -----+
            /                 \
... -> V5 --+                  +--> V7-合并版 [本地 master]
            \                 /
             +--> V5-注释版 --+
```
此方法真实保留历史, 但是历史线会变得复杂, 充满了各种合并提交.

* 方法2: Rebase
git rebase upstream/master 会执行如下操作:
找到 V5, 随后把超前于 V5 的所有提交剪切下来, 放入临时区域, master 分支会退回到 V5, 随后 master 分支移动到 upstream/master 的位置, 即 V6, 随后在 V6 的基础上应用一遍刚刚剪切下来的 V5-注释版, 得到 V6-注释版:

```
... -> V5 -> V6 [upstream/master] -> V6-注释版 [本地 master]
```
此方法改写了历史, V5-注释版会从历史中消失, 倘若有人拉取过 V5-注释版, 那么这是再 rebase 会造成很大的麻烦.

本次提交中, 我先 Fork 了 nanochat, 此时我的 master 分支是: Karpathy 的旧提交 -> 我的注释提交 (因为我是前一天晚上 git clone 的), 而远程的 origin/master 的提交历史是: Karpathy 的旧提交 -> Karpathy 的最新提交, 这就导致这两条历史线不是简单的前后关系, 而是从一个点走向两个不同的未来, 为了防止覆盖远程仓库可能很重要的提交, 所以 Git 选择了拒绝推送.

我下一步决定使用 git pull 来将远程的历史整合到本地, 结果遇到了第二个问题 -- Need to specify how to reconcile divergent branches. 检查到历史分岔的时候, Git 会给出两个选择, 到底是选择 Merge 还是 Rebase 来做合并, 此时我通过 git config --global pull.rebase false 来告诉 Git, 采用 merge 策略, 此时再次运行 git pull origin master, 由于 Git 知道了规则, 会将远程历史和本地历史做合并, 并创建一个新的合并提交. 这时再 git push origin master 就可以正常推送, 并保留了远程的所有内容.

```bash
git status
git add .
git commit -m "modified my README"

git fetch upstream
git merge upstream/master
git push origin master
```