import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Probability distribution dictionary
    prob_distrib = {}

    # If page has no outgoing links, then return all pages with equal probability
    if page not in corpus:
        for link in corpus:
            prob_distrib[link] = 1 / len(corpus)
        return prob_distrib

    for link in corpus:
        prob_distrib[link] = (1 - damping_factor) / len(corpus)

        if link in corpus[page]:
            prob_distrib[link] += damping_factor / len(corpus[page])

    return prob_distrib

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # initialize the dictionary with 0 page rank value for all pages
    dict_pages = {page: 0 for page in corpus}

    # select random page
    page = random.choice(list(corpus.keys()))
    dict_pages[page] += 1/n

    page = None

    for i in range(n-1):
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model.keys()), list(model.values()), k=1)[0]
        dict_pages[page] += 1/n

    return dict_pages

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(corpus)

    # initialize the dictionary with normalized page rank value for all pages
    page_rank_dict = {page: 1/total_pages for page in corpus}

    repeat = True
    while repeat:

        repeat = False
        for page in corpus:
            new_rank = (1 - damping_factor) / total_pages
            pr_sum = sum(page_rank_dict[key] / len(corpus[key]) for key in corpus if page in corpus[key])
            pr_sum += sum(page_rank_dict[key] / total_pages for key in corpus if page not in corpus[key])

            new_rank += damping_factor * pr_sum
            rank_diff = abs(page_rank_dict[page] - new_rank)
            # print(f"  {page}: old rank: {page_rank_dict[page]:.4f}: new rank: {new_rank:.4f}: diff: {rank_diff:.4f}")

            if rank_diff > 0.001:
                repeat = True

            page_rank_dict[page] = new_rank

    return page_rank_dict

if __name__ == "__main__":
    main()
