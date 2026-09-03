"""Public-domain excerpt of Shakespeare for character-level autoregressive training."""

import torch

from neural_blueprint.runtime.tokenizer import CharacterTokenizer

TINY_SHAKESPEARE = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country, he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a vice in him.
You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?
"""


def load_token_dataset(
    text: str,
    tokenizer: CharacterTokenizer,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sliding window (x, y) token prediction tensors from text."""
    ids = tokenizer.encode(text)
    if len(ids) <= block_size + 1:
        reps = (block_size + 2) // max(1, len(ids)) + 1
        ids = ids * reps

    t_ids = torch.tensor(ids, dtype=torch.long)
    num_samples = len(t_ids) - block_size
    x = torch.stack([t_ids[i : i + block_size] for i in range(num_samples)])
    y = torch.stack([t_ids[i + 1 : i + 1 + block_size] for i in range(num_samples)])
    return x, y
