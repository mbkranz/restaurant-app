#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[1]:


#sqlalchemy and pandas for data 
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd
#spacy for tokenization
from spacy.lang.en import English # Create the nlp object
import spacy
#gensim for similarity
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities.docsim import MatrixSimilarity,Similarity
#itertools for getting similarity edges
#networkx for organizing similarities
#plotly for visualization


# In[31]:


def tokenize_text(text_str,nlp_obj):
    '''
    use spacy to separate text into words
    (ie tokenization)
    and return the lemmatization 
    (ie feet for footing and foot)
    for only nouns and adjectives
    
    TODO: refine methodology
    '''
    spacy_doc = nlp_obj(text_str)
    
    tokenized_doc = [
        token.lemma_
        for token in spacy_doc
        if token.pos_ in ("NOUN","ADJ")
        ]
    
    return tokenized_doc
    #return spacy_doc
        


# ## Make dictionary & corpus

# In[32]:


nlp = spacy.load('en_core_web_sm')


# In[33]:


#TODO: combine reviews in SQL to scale
#TODO: sciktilearn?


# In[34]:


reviews_tokenized = (
    review_df
    #.head(2)
    .groupby('business_id')
    .text
    .apply(lambda x: ' '.join(x))
    .apply(tokenize_text,nlp_obj=nlp)
)


# In[35]:


reviews_dictionary = Dictionary(reviews_tokenized)


# In[36]:


reviews_dictionary.num_docs


# In[37]:


review_df.business_id.unique().shape


# In[38]:


#corpus
reviews_corpus = [reviews_dictionary.doc2bow(doc) for doc in reviews_tokenized]


# In[39]:


#tfidf with document being each restaurant and corpus being all restaurants
reviews_tfidf_model = TfidfModel(reviews_corpus)


# In[40]:


reviews_tfidf_docs = [reviews_tfidf_model[review] for review in reviews_corpus]


# ## Get Similarities

# In[3]:


# Function which returns subset or r length from n 
# https://docs.python.org/2/library/itertools.html
from itertools import combinations,groupby
import inspect


# In[46]:


def make_edges_df(node_list,
                  node_names=['x','y']):
    edges = combinations(node_list,2)
    edges_df = pd.DataFrame(edges,columns=node_names)
    return edges_df

def _filter_similarities(doc_indices,similarities_to_corpus):
    #get similarity for each pair
    similarities_to_group = {
        (doc1_index,doc2_index):similarities_to_corpus[doc2_index]
        for doc1_index,doc2_index in doc_indices.to_records(index=False)
    }
    
    series = pd.Series(similarities_to_group)
    series.index.names = ['docx','docy']
    return series

def get_tfidf_similarities(doc_index_series):
    tfidfs_from = reviews_tfidf_docs[doc_index_series.name]
    similarities_to = similarity_indices[tfidfs_from]
    return _filter_similarities(doc_index_series,similarities_to)


# In[47]:


#similarity indices for each doc
similarity_indices = MatrixSimilarity(reviews_tfidf_docs)


# In[48]:


#get index:name mappings
doc_mapping = (
    review_df
    .groupby('business_id')
    ['name']
    .apply(lambda x: x.unique()[0])
    .to_frame()
    .assign(doc_index=range(701))
)

#ie switch index to key and token str to object
token_mapping = {
    i:token 
    for token,i in reviews_dictionary.token2id.items()
} 


# In[49]:


doc_mapping.head(2)


# In[ ]:


#for orange and brew and bevande coffee (doc indices 0 and 172):
#doc_id = 0
all_bag_of_words_list = []
for doc_id in range(len(reviews_corpus)):
    bag_of_words = (
        pd.DataFrame(
            {"frequency":dict(reviews_corpus[doc_id]),
             "tf_idf":dict(reviews_tfidf_docs[doc_id]),
             "business":doc_mapping[doc_id],
             "word":token_mapping
            }
        )
        .set_index(['business','word'])
    )
    all_bag_of_words_list.append(bag_of_words)                                 


# In[ ]:


all_bag_of_words_df = pd.concat(all_bag_of_words_list)


# In[ ]:


all_similarities_list = []
for doc_id in range(len(reviews_corpus)):
    #doc example
    doc = reviews_corpus[doc_id]
    doc_tfidf = reviews_tfidf_model[doc]
    similarities = similarity_indices[doc_tfidf]
    
    all_similarities_list.append(similarities)    


# ## Make Network

# In[4]:


import networkx as nx


# In[5]:


node_attribute_names = [
    'node_for_adding',
    'business_name',
    'stars',
    'address',
    'review_count',
    'categories'
]

node_df = (
    business_info_df
    .set_index('business_id')
    .join(doc_mapping[['doc_index']])
    .sort_values('doc_index')
    .rename(columns={
        "doc_index":"node_for_adding",
        "name":"business_name"
    })
    [node_attribute_names]
)


# In[53]:


node_df.head(3)


# In[54]:


#edge_df = make_edges_df(node_df['node_for_adding'])


# In[55]:


tfidf_similarities = (
    make_edges_df(node_df['node_for_adding'])
    .groupby('x')
    .apply(get_tfidf_similarities)
    .reset_index('x',drop=True)
)


# In[58]:


graph = nx.Graph()


# In[59]:


#add nodes
#TODO: top word frequencies,location etc
#TODO: picture of restaurant and/or food for node
for i,node in node_df.iterrows():
    graph.add_node(**node.to_dict())


# In[60]:


#add edges
#TODO: add other attributes for hover etc like most similar words etc
#TODO: other similarity metrics
for nodes,tfidf in tfidf_similarities.iteritems():
    graph.add_edge(nodes[0],nodes[1],tfidf=tfidf)


# In[61]:


graph.number_of_edges()


# In[62]:


graph.number_of_nodes()


# In[1]:


repo_dir = '/Users/michaelkranz/Documents/restaurant-app/'
nx.write_gpickle(graph,f"{repo_dir}/data/champaign_restaurant_review.gpickle")


# ## get subgraph 
# - filtering (ie will be user defined)

# In[9]:


repo_dir = '/Users/michaelkranz/Documents/restaurant-app/'
graph = nx.read_gpickle(f"{repo_dir}/data/champaign_restaurant_review.gpickle")


# In[10]:


subgraph_edges = combinations(range(50),r=2)


# In[11]:


graph_sub = graph.edge_subgraph(subgraph_edges)


# In[50]:


len(graph_sub.edges())


# In[52]:


len(graph_sub.nodes())


# ## Visualizations

# ### Potential inspirations for design
# 
# #### Circos plots
# - [Cognitive ontology](https://www.ece.nus.edu.sg/stfpage/ybtt/papers/2014Brainmap/Interactive/index.html)
# 
# 
# #### Notes
# 
# - correlation matrix of selected restaurants
# - network Circos plot of selected restaurants
# - add input of relationship to terms typed to circos plot (or restaurant from st louis/chicago)?

# In[12]:


import plotly.express as px
import plotly.graph_objects as go


# In[13]:


spring_layout = nx.drawing.layout.spring_layout
graph_viz = graph_sub


# In[26]:


pos = spring_layout(graph_viz,weight='tfidf')

node_x = [x for x,y in pos.values()]
node_y = [y for x,y in pos.values()]
node_hover_text = [data for i,data in graph_viz.nodes(data='business_name')]


# In[240]:


#graph_viz.edges(data='tfidf')


# In[131]:


edge_x = []
edge_y = []
edge_mid_x = []
edge_mid_y = []
edge_tfidf = []
for node0,node1,tfidf in graph_viz.edges(data='tfidf'):
    
    #edge line viz coordinates
    x0, y0 = pos[node0]
    x1, y1 = pos[node1]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    
    #edge hover info trace coordinates
    x_mid = (x0+x1)/2dd
    y_mid = (y0+y1)/2
    edge_mid_x.append(x_mid)
    edge_mid_y.append(y_mid)
    
    #edge attribute info 
    edge_tfidf.append(str(tfidf))
    
    


# In[15]:


import numpy as np


# In[27]:


def compute_line_points(x0,x1,y0,y1,x_step=0.1):
    '''
    get several points on a line between two coordinates
    '''
    slope = (y1-y0)/(x1-x0)
    
    if x0>x1:
        x_step_sign = -1
    else:
        x_step_sign = 1
    
    y_intercept = y1-(slope*x1)   
    x_pts = np.arange(x0,x1, x_step_sign*x_step)
    y_pts = [(slope*x)+y_intercept for x in x_pts]
    #print(slope,y0,y1,x0,x1)
    return x_pts,y_pts


# In[28]:


edge_traces = []
for node0,node1,tfidf in graph_viz.edges(data='tfidf'):
    if tfidf>.1:
        #edge line viz coordinates
        x0, y0 = pos[node0]
        x1, y1 = pos[node1]

        #points
        x_pts,y_pts = compute_line_points(x0,x1,y0,y1)
        
        trace = go.Scatter(x=x_pts,
                           y=y_pts,
                           mode='lines',
                           line={'width':tfidf*20},
                           name=tfidf,
                           hovertext=tfidf)
        edge_traces.append(trace)
        


# In[29]:


compute_line_points(y0,y1,x0,x1)


# In[133]:


edge_hover_info = go.Scatter(
    #hover info here or any edge text attributes
    x=edge_mid_x,
    y=edge_mid_y,
    hovertext=edge_tfidf,
    mode='markers'
)


# In[135]:


edge_trace = go.Scatter(
    x=edge_x, 
    y=edge_y,
    #line=dict(width=edge_tfidf, color='#888'),
    #hoverinfo='none',
    #hovertext=edge_tfidf,
    
    #set size/width here
    mode='lines'
)


# In[30]:


node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    #hoverinfo='text',
    hovertext=node_hover_text,
    marker=dict(
        #showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        #colorscale='YlGnBu',
        reversescale=True,
        #color=node_hover_text,
        size=20,
        colorbar=dict(
            thickness=10,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))


# In[82]:


#pd.DataFrame(y for x,y in graph_viz.nodes(data=True)).categories.unique()


# In[83]:


#list(graph_viz.adjacency())


# In[ ]:


node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text


# In[137]:


fig = go.Figure(data=[edge_trace,edge_hover_info],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                #margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()


# In[19]:


traces = edge_traces
traces.append(node_trace)
fig = go.Figure(node_trace,
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                #hovermode='closest',
                 
                #margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()


# In[ ]:





# In[23]:


fig = go.Figure(node_trace,
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                #hovermode='closest',
                 
                #margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()


# In[24]:


fig = go.Figure(edge_traces,
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                #hovermode='closest',
                 
                #margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()


# In[31]:


traces = edge_traces
traces.append(node_trace)

fig = go.Figure(traces,
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                #hovermode='closest',
                 
                #margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()


# In[ ]:


#filter for restaurant category
#filter distance
#filter stars
#filter 


# In[ ]:


#ratings

#location

