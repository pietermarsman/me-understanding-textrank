import requests

def validate_wiki_response(response):
    if 'query' not in response:
        raise AttributeError('Not a valid query')
    if '-1' in response['query']['pages']:
        normalized_query = response['query']['pages']['-1']['title']
        raise KeyError('Could not find page "{}" on wikipedia'.format(normalized_query))
    

def query_wikipedia(page, language='en'):
    params = {
        'action': 'query',
        'format': 'json',
        'titles': page,
        'prop': 'extracts',
        'explaintext': True
    }
    response = requests.get('https://{}.wikipedia.org/w/api.php'.format(language), params=params).json()
    validate_wiki_response(response)
    
    text = next(iter(response['query']['pages'].values()))['extract']
    return text