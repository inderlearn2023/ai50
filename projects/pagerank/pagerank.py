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

    page_link_count = len(corpus[page])

    # If page has no outgoing links, then return all pages with equal probability
    if page_link_count == 0:
        for page in corpus:
            prob_distrib[page] = 1 / len(corpus)
        return prob_distrib

    for page in corpus:
        prob_distrib[page] = (1 - damping_factor) / len(corpus)

    for link in corpus[page]:
        prob_distrib[link] += damping_factor / page_link_count

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
    dict_pages = {}
    for link in corpus:
        dict_pages[link] = 0

    # Calculate PageRank values for each page by sampling `n` pages
    # according to transition model, starting with a page at random.
    page = random.choice(corpus.keys())

    for i in range(1, n):
        page_prob = transition_model(corpus, page, damping_factor)
        random_page = random.choices(list(page_prob.keys()), list(page_prob.values()))
        dict_pages[random_page[0]] += 1 / n

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

    page_rank_dict = {}
    for page in corpus:
        page_rank_dict[page] = 1/total_pages

    page_count = 0
    while page_count < len(corpus):
        new_rank = (1 - damping_factor) / total_pages


        for page in corpus:
            page_rank_dict[page] = (1 - damping_factor) / len(corpus)
            pr_sum = sum(page_rank_dict[key] / len(corpus[key]) for key in corpus if page in corpus[key])

            page_rank_dict[page] += damping_factor * pr_sum

            if abs(page_rank_dict[page] - new_rank) < 0.001:
                page_count += 1

    return page_rank_dict


if __name__ == "__main__":
    main()
