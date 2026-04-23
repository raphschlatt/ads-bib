# Search & Query Design

The quality of the final topic model and citation networks starts with the ADS
query. `ads-bib` does not invent the corpus for you; it turns the corpus you
ask for into a translated, topic-modeled, citation-ready dataset.

The ADS API supports a rich
[search syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax). Start
small, validate the result set, then expand to your full bibliometric question.

!!! note "Syntax errors come back from the ADS API, not from `ads-bib`"
    The query you set in `search.query` is passed to the ADS API unchanged.
    `ads-bib` does not parse or validate it. Malformed queries surface as an
    HTTP 400 from ADS during the `search` stage. Test a narrow query first,
    then scale up.

## The Basic Building Blocks

| Building block | Purpose | Example |
| --- | --- | --- |
| Seed set | Core publications by an author, group, or topic | `author:"Hawking, S*"` |
| Forward citations | Papers citing your seed set | `citations(author:"Hawking, S*")` |
| Backward references | Papers cited by your seed set | `references(author:"Hawking, S*")` |
| Topic filter | Restrict to a thematic subset | `abs:"black hole"` |
| Time filter | Restrict by publication date | `year:1974-1990` |
| Venue filter | Restrict to journals or collections | `bibstem:PhRvD` |

## Syntax Basics

The five rules below cover the situations where a query looks obvious but
returns a very different result set than intended. Each rule matches one real
ADS behavior; the external
[ADS syntax reference](https://ui.adsabs.harvard.edu/help/search/search-syntax)
is the full grammar.

### Phrases

Quotes turn a sequence of words into a single phrase. Without quotes, ADS
treats the tokens as independent terms and may match documents that contain
them in any order or not adjacent.

```text
abs:"black hole"      # matches the phrase
abs:black hole        # matches abs:black anywhere + the bare token hole
```

!!! warning "Quoting changes the result set by orders of magnitude"
    `abs:"black hole"` and `abs:black hole` are **not** synonyms. Use quotes
    whenever you mean a phrase; omit them only when you intentionally want
    independent term matches.

### Wildcards

A trailing `*` expands to any number of characters. ADS wildcards are
**trailing-only** — no leading wildcards, no middle wildcards, no glob or
regex semantics.

```text
author:"Hawking, S*"   # Hawking, S / Hawking, Stephen / Hawking, Susan / ...
author:"*awking, S"    # NOT supported
author:"H?wking, S"    # NOT supported (no single-character wildcard)
```

!!! warning "Wildcards are trailing-only"
    Leading-wildcard, middle-wildcard, regex, and glob syntax all fail on
    ADS. If you need more flexibility, enumerate alternatives with `OR`.

### Boolean operators

`AND`, `OR`, and `NOT` combine clauses. `AND` has higher precedence than `OR`,
so `A OR B AND C` is read as `A OR (B AND C)`. Use parentheses whenever the
reading order matters.

```text
author:"Hawking, S*" OR citations(author:"Hawking, S*")
(author:"Hawking, S*" OR author:"Penrose, R*") AND abs:"black hole"
```

!!! warning "`AND` binds tighter than `OR`"
    `A OR B AND C` is parsed as `A OR (B AND C)`. If you meant
    `(A OR B) AND C`, write it that way explicitly. The difference between
    the two often shows up as a 10×–100× change in result count.

### Negation

Exclude a term with `-` or `NOT`. Both prefixes work.

```text
author:"Hawking, S*" -abs:"cosmology"
author:"Hawking, S*" AND NOT abs:"cosmology"
```

### Common field prefixes

| Prefix | Scope |
| --- | --- |
| `author:` | Author names (supports trailing `*`) |
| `title:` | Paper title |
| `abs:` | Abstract |
| `full:` | Full text where available |
| `year:` | Publication year; supports `year:1974-1990` ranges |
| `bibstem:` | Journal / collection code (e.g. `PhRvD`, `ApJ`) |
| `citations(...)` | Forward citations of the inner query |
| `references(...)` | Backward references of the inner query |

`full:` is the broadest (and noisiest) prefix; prefer `abs:` or `title:` for
topic filters when precision matters.

## Common Query Patterns

### 1. One author or one seed library

```yaml
search:
  query: 'author:"Hawking, S*"'
```

Use this when you want the direct publication set first and will decide later
whether to widen the network.

### 2. Seed set plus topic filter

```yaml
search:
  query: '(author:"Hawking, S*") AND abs:"black hole"'
```

Use this when an author has a broad oeuvre but you only want one topic area.

### 3. Seed set plus forward citations

```yaml
search:
  query: 'author:"Hawking, S*" OR citations(author:"Hawking, S*")'
```

Use this when you want the reception history or downstream influence of a seed
library.

### 4. Topic-driven field slice

```yaml
search:
  query: '(abs:"quantum gravity" OR title:"quantum gravity") AND year:1980-2005'
```

Use this when the corpus is defined by a topic rather than by authorship.

!!! warning "Start narrow"
    A first-run query that resolves to more than ~10,000 ADS records will
    spend most of its time in export and translation. Validate the pipeline
    end-to-end with a small slice before you scale up.

## A Good Iteration Pattern

1. Start with a narrow query.
2. Run the pipeline once and inspect:
   - publication count (see `counts.total_processing.publications` in
     [`run_summary.yaml`](outputs.md#run-summaryyaml))
   - reference count
   - topic coherence
   - citation network density
3. Widen the query only after the small run looks sensible.

If you change the query, rerun `search` and `export`. If you only tune later
topic-model steps, keep the search result fixed and iterate from `topic_fit`
onward.

## Search Refresh Flags

Use the refresh flags deliberately:

- `search.refresh_search=true`
  - reruns the ADS search itself
- `search.refresh_export=true`
  - reruns the bibcode-to-metadata export

During topic-model tuning on the same dataset, leave both `false` so you do not
re-hit the ADS API unnecessarily.

## Practical Advice

- Prefer smaller, intelligible query slices over one giant first run.
- Mix author, citation, and keyword clauses instead of relying on one field.
- Validate counts before you interpret topic structure.
- If a corpus looks noisy, the problem is often the query, not the clustering.

## Read next

- [Topic Modeling](topic-modeling.md) — once the query size looks right
- [Pipeline Guide](pipeline-guide.md) — stage order and when to re-run
- [Get Started](get-started.md#run-the-cli) — minimal CLI reminder
