# Search & Query Design

The quality of the final topic model and citation networks starts with the ADS
query. `ads-bib` does not invent the corpus for you; it turns the corpus you
ask for into a translated, topic-modeled, citation-ready dataset.

The ADS API supports a rich
[search syntax](https://ui.adsabs.harvard.edu/help/search/search-syntax). Start
small, validate the result set, then expand to your full bibliometric question.

## The Basic Building Blocks

| Building block | Purpose | Example |
| --- | --- | --- |
| Seed set | Core publications by an author, group, or topic | `author:"Hawking, S*"` |
| Forward citations | Papers citing your seed set | `citations(author:"Hawking, S*")` |
| Backward references | Papers cited by your seed set | `references(author:"Hawking, S*")` |
| Topic filter | Restrict to a thematic subset | `abs:"black hole"` |
| Time filter | Restrict by publication date | `year:1974-1990` |
| Venue filter | Restrict to journals or collections | `bibstem:PhRvD` |

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

When you are happy with the dataset definition, continue with
[Topic Modeling](topic-modeling.md) and the deeper tuning advice in the
[Pipeline Guide](pipeline-guide.md).
