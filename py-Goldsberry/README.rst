py-Goldsberry
=============

A Python Package for easily acquiring NBA Data for analysis

What is ``py-Goldsberry``?
--------------------------

``py-Goldsberry`` is designed to give the user easy access to data
available from stats.nba.com in a form that facilitates innovative
analysis. With a few simple commands, you can have access to virtually
any data available on the site in an easy to analyze format. In fact,
some of the data is in a less summarize form giving you the opportunity
to work with the most raw data possible when you are attempting to
answer questions that interest you.

Why was it built?
-----------------

I attended the 2015 Sloan Sports Analytics conference and had the
fortunate opportunity to listen to `Kirk Goldsberry`_ address the crowd
regarding the state of analytics in sports (You can watch the talk
`here`_). One of the questions he addressed at the end was related to
the availability of data (or lack thereof in some instances). Basically,
he concluded that the lack of availability of some of the newest data is
actually hindering the progression of analytics in sports. Innovation is
now restricted to those with access to data instead of to the entire
community of interested parties. I wrote (am writing) this package in an
attempt to help address this issue in whatever small way I can.

This package is a work in progress. As the NBA continues to make more
data available, I will do my best to update ``py-Goldsberry`` to reflect
these additions. Currently, there is almost a cumbersome amount of data
available from the NBA so dealing with what is there is a bit of a
challenge. 

*UPDATE:* The NBA has apparently masked some of the tables that were previously available. The log level data is no longer available. This is disappointing as there was a multitude of research opportunities availble with the use of the data. Hopefully, the NBA will make this data available again in the near future.

Getting started
---------------

To get started with ``py-Goldsberry``, you need to install and load the
package. From your terminal, run the following command:

::

    pip install py-goldsberry

Once you have the package installed, you can load it into a Python
session with the following command:

.. code:: python

    import goldsberry
    import pandas as pd

The package is designed to work with `pandas`_ in that the output of
each API call to the NBA website it returned in a format that is easily
converted into a pandas dataframe.

Getting a List of Players
~~~~~~~~~~~~~~~~~~~~~~~~~

One of the key variables necessary to fully utilize ``py-Goldsberry`` is
``playerid``. This is the unique id number assigned to each player by
the NBA. ``py-Goldsberry`` has a top-level class ``PlayerList()``
built-in to give you quick access to a list of players and numbers. 

.. code:: python


    players2010 = goldsberry.PlayerList(Season='2010-11')
    players2010 = pd.DataFrame(players2010.players())
    players2010.head()

If you want a list of every game during the current season use the ``GameIDs()`` class:

.. code:: python

    games = goldsberry.GameIDs()
    games = pd.DataFrame(games.game_list())
    games.head()

As you get started with ``py-goldsberry``, TAB completion in either Jupyter or IPython is going to be your best friend. I'm working on documetation, but there is a great deal of it to do and I don't have that much time. 

.. _Kirk Goldsberry: http://twitter.com/kirkgoldsberry
.. _here: https://www.youtube.com/watch?v=wLf2hLHlFI8
.. _pandas: http://pandas.pydata.org/
