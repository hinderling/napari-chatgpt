import subprocess
import sys
import traceback
import difflib
import re
from bs4 import BeautifulSoup
from functools import cache
from subprocess import CalledProcessError
from typing import List

from arbol import aprint, asection
from langchain import LLMChain, PromptTemplate
from langchain.callbacks import StdOutCallbackHandler, CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM
import requests

from napari_chatgpt.utils.openai_key import set_openai_key

_pip_install_missing_prompt = f"""
Task:
You competently write the 'pip install <list_of_packages>' command required to run the following python {sys.version.split()[0]} code:

CODE:
#______________
{'{input}'}
#______________

Only list packages that are ABSOLUTELY necessary, NO other packages should be included in the list.
Mandatory dependencies of packages listed should not be included.
Answer should be a space-delimited list of packages (<list_of_packages>) without text or explanations before or after.
ANSWER:
"""


def required_libraries(code: str, llm: BaseLLM = None):
    # Cleanup code:
    code = code.strip()

    # If code is empty, nothing is missing!
    if len(code) == 0:
        return []

    # Ensure that OpenAI key is set:
    set_openai_key()

    # Instantiates LLM if needed:
    llm = llm or ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

    # Make prompt template:
    prompt_template = PromptTemplate(template=_pip_install_missing_prompt,
                                     input_variables=["input"])

    # Instantiate chain:
    chain = LLMChain(
        prompt=prompt_template,
        llm=llm,
        verbose=True,
        callback_manager=CallbackManager([StdOutCallbackHandler()])
    )

    # Variable for prompt:
    variables = {"input": code}

    # call LLM:
    list_of_packages_str = chain(variables)['text']

    # Parse the list:
    list_of_packages = list_of_packages_str.split()

    return list_of_packages


def validate_package_name(package_name):
    """Find the most relevant package name for a given package name using PyPI search.
    This helps to work around cases where the library name is not exactly the same as the PyPI package name, e.g. useq and useq-schema."""
    
    search_url = f"https://pypi.org/search/?q={package_name}"
    response = requests.get(search_url)
    if response.status_code == 200:

        #get all links from the searched page
        soup = BeautifulSoup(response.content, "html.parser")
        package_list = [link.get("href") for link in soup.find_all("a",href=True)]

        #find links that match the format /project/<any_package>/ and return [<any_package_name>]
        regex = re.compile(r".*/project/(.*)")
        matches = [regex.match(string).group(1)[:-1] for string in package_list if regex.match(string)]

        #find the closest match to the package name
        closest_match = difflib.get_close_matches(package_name, matches, n=1,cutoff=0)
        print(closest_match)
        if closest_match ==[]:
            return None
        else:
            return closest_match[0]

    else:
        #Could not access PyPI search
        print(f"Could not find package name for '{package_name}'")
        return None



def pip_install(packages: List[str], ignore_obvious: bool = True) -> bool:
    if ignore_obvious:
        packages = [p for p in packages if
                    not p in ['numpy', 'napari', 'magicgui', 'scikit-image','useq-schema']]

    try:
        with asection(f"Installing {len(packages)} packages with pip:"):
            for package in packages:
                package = validate_package_name(package)
                _pip_install_single_package(package)
        return True
    except CalledProcessError:
        aprint(traceback.format_exc())
        return False



@cache
def _pip_install_single_package(package):
    aprint(f"Pip installing package: {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    aprint(f"Installed!")
