# The plan is that we will process:
# 1) The films of the normal saga from https://starwars.fandom.com/wiki/Star_Wars_saga
# 2) The films from the categories
# 3) Get the pages of the characters appearing in the films or movies

import requests
from bs4 import BeautifulSoup
import re
import os

# Static links to the web pages containing list of specialized links for films, series, characters, planets.
# (the mandalorian tv serie is treated differently by wikipedia, so we're going to handle that as well)
film_link = "https://en.wikipedia.org/wiki/List_of_Marvel_Cinematic_Universe_films"
characters_link = "https://en.wikipedia.org/wiki/Category:Marvel_Comics_superheroes"
actor_link = "https://en.wikipedia.org/wiki/List_of_Marvel_Cinematic_Universe_film_actors"
# planets_link = "https://en.wikipedia.org/wiki/List_of_Star_Wars_planets_and_moons"

def convert_page_to_text(html_tag, title_tags, keep_headers, headers, page_title):
    """
    This function converts the html_tag with the page content into text. The text will be split according to the header tags (e.g., h1, h2, etc).
    The result is another html page but with removed unnecessary elements and nested structure. There will only be the tags in title_tags and <p> for the paragraph content

    :param html_tag: the tag with the page content
    :param title_tags: The tags associated with section headers (e.g. h1, h2, h3, etc.)
    :param keep_headers: True if you want to keep the paragraph, False if ignore the headers provided in the parameter headers
    :param headers: titles of the section to be ignored (e.g Bibliography, Notes, etc.)
    :return: page converted into an html file, but without noise
    """

    regex_headers = ("|".join(headers))
    # Unwrap meta tags. Some pages have data in a format like <meta>...</meta>. We need to extract elements from those.
    meta_tags = html_tag.find_all("meta")
    for meta_tag in meta_tags:
        meta_tag.unwrap()

    full_page_text = ""
    last_tag_name = None
    ignore_paragraph = False

    # Write the title of the page for future chunk generation
    full_page_text += f"<h1>{page_title}</h1>"

    # Removing span tags with class IPA --> These contain transcripts in the international phonetic alphabet for the pronounciation of characters' names.
    # These characters have problems when being saved into a file.
    for tag in html_tag.find_all("span", {"class": "IPA"}):
        tag.decompose()

    # Go through the children and extract all the paragraphs in a unique <p> tag until a tag inside title_tags is found. Then, close the p tag and add the new
    # header section.
    for child_tag in html_tag.children:
        if child_tag.name == "p":
            # Checking if the paragraph is inside a paragraph to be ignored (the ones with header inside headers)
            if not ignore_paragraph:
                # In the case the last tag was not a paragraph, add the start of a paragraph tag <p>
                if last_tag_name is None or last_tag_name != "p":
                    full_page_text += "<p>"

                full_page_text += child_tag.text
                last_tag_name = "p"
        else:
            # If now we're reading a tag that is not p and the last tag was p, close the tag.
            if last_tag_name is not None and last_tag_name == "p":
                full_page_text += "</p>"
            last_tag_name = child_tag.name

            # If the header of the paragraph is to be ignored, just skip it.
            if child_tag.name == "h2" and re.search(regex_headers, child_tag.text, flags=re.IGNORECASE):
                ignore_paragraph = not keep_headers
            elif child_tag.name == "h2":
                ignore_paragraph = keep_headers

            # Keep the header tag
            if not ignore_paragraph and child_tag.name in title_tags:
                full_page_text += f'<{child_tag.name}>{child_tag.text}</{child_tag.name}>'

    # Remove [] as citation to the notes
    full_page_text = re.sub("\[.+?\]", "", full_page_text)
    # Remove special &...; characters, special escape chars in HTML
    full_page_text = re.sub("&(.)+;", "", full_page_text)
    # Remove transcription in Japanese of the characters
    full_page_text = re.sub("\((.)*Japanese(.)*\)", "", full_page_text)

    return full_page_text


def save_pages(web_pages_texts, file_names, directory):
    """
    Creates a .html file for each extracted element.
    :param web_pages_texts: texts of the pages
    :param file_names: names to be used for the files. This should be the title of the page as well
    :param directory: directory where to store the data. If not existing, it will be created
    """
    for file_name, text in zip(file_names, web_pages_texts):
        file_name = directory + file_name + ".html"

        #Create directory, if it doesn't exist already
        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(file_name, "w", encoding='utf-8') as html_file:
            html_file.write(text)

def wikipedia_extract_link(content_tag, link_name_regex):
    """
    Extract the wikipedia links inside the page that redirect to related pages. This can be used for accessing the single pages from
    the page containing a list of links. (e.g. List of characters -> Retrieve links to all single characters)
    :param content_tag: html tag with the relevant content
    :param link_name_regex: regex used for assessing if the text of the link is relevant.
    :return: returns links to related pages
    """
    # retrieve a tags and the link inside it
    links = content_tag.find_all("a")
    # Generate a dict with <text link> as key and <html tag> as value
    link_dict = {tag.text: tag.get("href") for tag in links}

    compiled_wiki_regex = re.compile("wiki")
    remove_key = []

    # Get the keys from the dictionary that are not referring to wikipedia pages
    for key, link in link_dict.items():
        if link is None or not re.search(compiled_wiki_regex, link):
            remove_key.append(key)

    # Remove the links not related to wiki pages
    for k in remove_key:
        del link_dict[k]

    relevant_links = []

    # keep only the links related to relevant pages that satisfy the link_name_regex
    for key, value in link_dict.items():
        if re.search(link_name_regex, key):
            relevant_links.append(value)

    # remove text after the hash
    for idx, link in enumerate(relevant_links):
        relevant_links[idx] = re.sub("#(.+)$", "", link)

    # Remove duplicates
    relevant_links = list(set(relevant_links))

    # Add the wikipedia prefix to have an absolute link
    for idx, link in enumerate(relevant_links):
        relevant_links[idx] = "https://en.wikipedia.org/" + link

    return relevant_links


def extract_page_content(link_list, keep_headers_bool, headers, title_tags):
    """
    Extracts page texts and related file names to be saved later
    :param link_list: list of the links to the web pages to scrape
    :param keep_headers_bool: whether the headers provided should be kept or not
    :param headers: which headers should be kept or discarded
    :param title_tags: html tags associated to title
    :return: tuple (text of the web pages refined, names to be used for saving the files)
    """
    web_pages_texts = []
    file_names = []

    for link in link_list:
        #Get the page
        page = requests.get(link)
        soup = BeautifulSoup(page.content, "html.parser")
        #retrieve the content of the page and the title
        div_content = soup.find("div", {"class": "mw-content-ltr mw-parser-output"})
        page_title = soup.find("h1", {"id": "firstHeading"}).text

        # convert the pages into text
        web_pages_texts.append(convert_page_to_text(div_content, title_tags, keep_headers_bool, headers, page_title))

        # Remove the first page of the wikipedia link to get the title of the page
        file_name = re.findall("/wiki/(.+)", link)[0]
        file_name = re.sub(":", "", file_name)
        file_names.append(file_name)

    return web_pages_texts, file_names


def export_movies():
    """
    Saves movie files scraped from wikipedia
    """
    # Get the page and convert it into beautiful soup
    film_page = requests.get(film_link)
    soup_film = BeautifulSoup(film_page.content, "html.parser")

    # retrieve the page content
    div_content = soup_film.find("div", {"class": "mw-content-ltr mw-parser-output"})

    # title tags are these at the beginning of this section
    title_tags = ["h1", "h2", "h3", "h4", "h5"]
    # headers to ignore because they don't have relevant info
    ignore_headers = ["Reception", "Unproduced and abandoned projects", "Documentaries", "Notes", "See also",
                      "References", "External links"]

    # extract the text of the page
    full_page_text = convert_page_to_text(div_content, title_tags, False, ignore_headers, "List of Star Wars films")

    # Save the pages
    save_pages([full_page_text], ["Movies"], directory="./web_pages/")

    # Extract links of the children pages
    film_title_regex = re.compile("Episode|Rogue One|The Clone Wars|^Solo: A Star Wars Story$", flags=re.IGNORECASE)
    relevant_links = wikipedia_extract_link(div_content, film_title_regex)

    keep_headers = ["Plot", "Cast"]

    web_pages_texts, file_names = extract_page_content(relevant_links, True, keep_headers, title_tags)

    save_pages(web_pages_texts, file_names, directory="./web_pages/")


def wikipedia_extract_character_link(html_tag):
    """
    Extract links of children pages of the character list
    :param html_tag: html tag containing the links
    :return: extracted links
    """
    div_link_container = html_tag.find_all("div", {"role": "note"})

    links = []
    for div in div_link_container:
        link_to_resource = div.find("a")["href"]
        full_link = "https://en.wikipedia.org/" + link_to_resource
        links.append(full_link)

    return links


def export_characters():
    """
    Saves the pages containing the information concerning the characters
    """
    characters_page = requests.get(characters_link)
    soup_chars = BeautifulSoup(characters_page.content, "html.parser")

    div_content = soup_chars.find("div", {"class": "mw-content-ltr mw-parser-output"})

    title_tags = ["h1", "h2", "h3", "h4", "h5"]
    ignore_headers = ["References", "External Links"]

    full_page_text = convert_page_to_text(div_content, title_tags, False, ignore_headers, "List of Star Wars characters")

    save_pages([full_page_text], ["Characters"], directory="./web_pages/")

    relevant_links = wikipedia_extract_character_link(div_content)

    web_pages_texts, file_names = extract_page_content(relevant_links, False, [], title_tags)

    save_pages(web_pages_texts, file_names, directory="./web_pages/")


def convert_page_series_to_text(html_tag, page_title):
    # Remove characters associated to pronunciation that results in problems when saving the file
    for tag in html_tag.find_all("span", {"class": "IPA"}):
        tag.decompose()

    full_page_text = ""

    full_page_text += f"<h1>{page_title}</h1>"
    full_page_text += "<p>"

    for child_tag in html_tag.children:
        if child_tag.name == "h2":
            full_page_text += "</p>"
            break

        if child_tag.name == "p":
            full_page_text += child_tag.text

    # the title of the episode is inside the td tag with class "summary"
    episode_titles = html_tag.find_all("td", {"class": "summary"})
    # the summaries of the episodes are contained in the td tag with class "description"
    episode_summaries = html_tag.find_all("td", {"class": "description"})

    # For each title and summary, create a web page with <title> tag and <p> with the related text
    for title, summary in zip(episode_titles, episode_summaries):
        title_tag = f"<title>{title.text}</title>"
        summary_tag = f"<p>{summary.text}</p>"
        full_page_text += title_tag + summary_tag

    # Remove [] as citation to the notes
    full_page_text = re.sub("\[.+?\]", "", full_page_text)
    # Remove special &...; characters, special escape chars in HTML
    full_page_text = re.sub("&(.)+;","", full_page_text)
    # Remove strange pronounciation of the characters
    full_page_text = re.sub("\((.)*Japanese(.)*\)", "", full_page_text)

    return full_page_text


def extract_actors_content(links):
    pages_texts = []
    file_names = []

    for link in links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, "html.parser")

        div_content = soup.find("div", {"class": "mw-content-ltr mw-parser-output"})
        page_title = soup.find("h1", {"id": "firstHeading"}).text

        page_text = convert_page_series_to_text(div_content, page_title)
        pages_texts.append(page_text)

        file_name = re.findall("/wiki/(.+)", link)[0]
        file_name = re.sub(":", "", file_name)
        file_names.append(file_name)

    return pages_texts, file_names


def export_actors():
    """
    Saves the pages containing the data of the series.
    """
    series_page = requests.get(actor_link)
    soup_series = BeautifulSoup(series_page.content, "html.parser")

    div_content = soup_series.find("div", {"class": "mw-content-ltr mw-parser-output"})

    div_link_container = div_content.find_all("div", {"role": "note"})

    links = []
    for div in div_link_container:
        link_to_resource = div.find("a")

        if link_to_resource is None:
            continue

        link_to_resource = link_to_resource["href"]
        full_link = "https://en.wikipedia.org/" + link_to_resource
        links.append(full_link)

    # The Mandalorian is treated differently as, for every season there is an additional wikipedia page.
    # We need to treat it differently
    # mandalorian_links = retrieve_mandalorian_links()
    # links.extend(mandalorian_links)

    web_pages_texts, file_names = extract_actors_content(links)

    save_pages(web_pages_texts, file_names, directory="./web_pages/")

# def retrieve_mandalorian_links():
#     mandalorian_page = requests.get(mandalorian_link)
#     soup_mandalorian = BeautifulSoup(mandalorian_page.content, "html.parser")
#     div_content = soup_mandalorian.find("div", {"class": "mw-content-ltr mw-parser-output"})
#     links_to_mandalorian_seasons = div_content.find_all("a", {"title": re.compile("^The Mandalorian season*")})
#     links_to_mandalorian_seasons = [link["href"] for link in links_to_mandalorian_seasons]

#     after_hash_regex = re.compile("#(.)*")
#     filtered_links_to_mandalorian = [re.sub(after_hash_regex, "", link) for link in links_to_mandalorian_seasons]

#     filtered_links_to_mandalorian = list(set(filtered_links_to_mandalorian))

#     filtered_links_to_mandalorian = ["https://en.wikipedia.org" + link for link in filtered_links_to_mandalorian]

#     return filtered_links_to_mandalorian


# def export_planets():
#     """
#     Saves pages containing the data of the planets
#     """
#     planets_page = requests.get(planets_link)
#     soup_planets = BeautifulSoup(planets_page.content, "html.parser")

#     div_content = soup_planets.find("div", {"class": "mw-content-ltr mw-parser-output"})

#     file_names = []
#     text_pages = []

#     astrography_text = "<h1>Canon Astrography of Star Wars</h1>"
#     h2_astrography = div_content.find("span", {"id": "Star_Wars_canon_astrography"}).parent
#     next_tag = h2_astrography.next_sibling
#     while next_tag.name != "h2":
#         if next_tag.name == "p" or next_tag.name == "ul" or next_tag.name == "dl":
#             astrography_text += next_tag.text

#         next_tag = next_tag.next_sibling

#     file_names.append("Astrography")
#     text_pages.append(astrography_text)

#     new_line_regex = re.compile("\n")

#     for row in div_content.find_all("table")[0].find_all("tr")[1:]:
#         if len(row.find_all("td")) < 5:
#             continue
#         # Access to the first and fifth column containing the name of the planet and a small description, respectively.
#         planet_name_col = row.find_all("td")[0]
#         descr_col = row.find_all("td")[4]
#         filtered_name = re.sub(new_line_regex, "", planet_name_col.text)
#         filtered_descr = re.sub(new_line_regex, "", descr_col.text)

#         text_page = f"<h1>{filtered_name}</h1><p>{filtered_descr}</p>"
#         text_pages.append(text_page)
#         file_names.append(filtered_name)

#     save_pages(text_pages, file_names, directory="./web_pages/")


if __name__ == "__main__":
    # export_planets()
    export_actors()
    export_movies()
    export_characters()