# `Parsnip`

## 🥕 `argparse` with conditional argument groups.

As `minimax.train` is the single point-of-entry for training, its command-line arguments can grow quickly in number with each additional autocurriculum method supported in `minimax`. This complexity arises for several reasons:

- New components in the form of training runners, environments, agents, and models may require additional arguments
- New components may require existing arguments shared with previous components
- New components may overload the meaning of existing arguments used by other components

We make use of a custom module called `Parsnip` to help manage the complexity of specifying and parsing command-line arguments. `Parsnip` allows the creation of named argument groups, which allows adding new arguments while explicitly separating them into name spaces. Each argument group results in its own kwarg dictionary when parsed. 

`Parsnip` directly builds on `argparse` by adding the notion of a "subparser". Here, a subparser is simply an `argparse` parser responsible for a named argument group. Subparsers enable some useful behavior:
- Arguments can be added to the top-level `Parsnip` parser or to a subparser. 
- Each subparser is initialized with a `name` for its corresponding argument group. All arguments under this subparser will be contained in a nested kwarg dictionary under the key equal to `name`. 
- Each subparser can be initialized with an optional `prefix`, in which case all command-line arguments added to the subparser will be prepended with the value of `prefix` (see example below), thus creating a namespace for the corresponding argument group.
- Subparsers can be added conditionally, based on the specific value of a top-level argument (with support for the wildcard `*`).
- After parsing, `Parsnip` produces a kwargs dictionary containing a key:value pair for each top-level argument and a nested kwargs dictionary, under the key `<prefix>` containing the parsed arguments managed by each active subparser initialized with `prefix=<prefix>`.

Other than these details, `Parsnip`'s interface remains identical to that of `argparse`. 

## A minimal example
In this example, we assume the parser is used inside a script called `run.py`.

```python
from util.parsnip import Parsnip

# Create a new Parsnip parser
parser = Parsnip()

# Add some top-level arguments (same as argparse)
parser.add_argument(
    '--name', 
    type=str,  
    help='Name of my farm.')
parser.add_argument(
    '--kind', 
    type=str,
    choices=['apple', 'radish'],
    help='What kind of farm I run.')
parser.add_argument(
    '--n_acres', 
    type=str,  
    help='Size of my farm in acres.')

# Create a nested argument group with a prefix
crop_subparser = parser.add_subparser(name='crop', prefix='crop')
parser.add_argument(
    '--n_acres', 
    type=str,  
    help='Size of land for growing radish, in acres.')

# Create a conditional argument group
radish_subparser = parser.add_subparser(
    name='radish',
    prefix='radish',
    dependency={'crop': 'radish'},
    dest='crop')
radish_subparser.add_argument(
    '--is_pickled'
    type=str2bool,
    default=False,
    help='Whether my farm produces pickled radish.')

# Create another conditional argument group
apple_subparser = parser.add_subparser(
    name='apple',
    prefix='apple',
    dependency={'crop': 'apple'},
    dest='crop')
apple_subparser.add_argument(
    '--kind'
    type=str,
    choices=['fuji', 'mcintosh'],
    default='fuji',
    help='Whether my farm produces pickled radish.')

args = parser.parse_args()
```

Then running this command

```bash
python run.py \
--name 'Radelicious Farms' \
--kind radish \
--n_acres 200 \
--crop_n_acres 150 \
--radish_is_pickled
```

would produce this kwargs dictionary:

```python
{
    'name': 'Radelicious Farms',
    'kind': 'radish',
    'n_acres': 200,
    'crop_args': {
        'n_acres': 150,
        'is_pickled': True
    }
}
```

Notice how the `prefix` for each subparser is appended to each argument name added to that subparser (e.g. `n_acres` became `crop_n_acres`, and `is_pickled` became `radish_is_pickled`). Also notice how the `radish_is_pickled` argument became active, as its activation conditions on `kind=radish`, as we specified when defining the `radish_subparser`.

Likewise, running this argument

```bash
python run.py \
--name 'Appledores Farms' \
--kind apple \
--n_acres 200 \
--crop_n_acres 150 \
--apple_kind fuji
```

results in this kwargs dictionary:

```python
{
    'name': 'Appledores Farms',
    'kind': 'apple',
    'n_acres': 200,
    'crop_args': {
        'n_acres': 150,
        'kind': 'fuji'
    }
}
```