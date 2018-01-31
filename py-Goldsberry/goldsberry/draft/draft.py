from goldsberry.masterclass import NbaDataProvider
from goldsberry.apiparams import *


class anthro(NbaDataProvider):
    def __init__(self, **kwargs):
        url_modifier = 'draftcombineplayeranthro'
        NbaDataProvider.__init__(self, url_modifier=url_modifier, default_params=p_draft, **kwargs)

    def data(self):
        return self._get_table_from_data(self._data_tables, 0)


class agility(NbaDataProvider):
    def __init__(self, **kwargs):
        url_modifier = 'draftcombinedrillresults'
        NbaDataProvider.__init__(self, url_modifier=url_modifier, default_params=p_draft, **kwargs)

    def data(self):
        return self._get_table_from_data(self._data_tables, 0)


class non_stationary_shooting(NbaDataProvider):
    def __init__(self, **kwargs):
        url_modifier = 'draftcombinenonstationaryshooting'
        NbaDataProvider.__init__(self, url_modifier=url_modifier, default_params=p_draft, **kwargs)

    def data(self):
        return self._get_table_from_data(self._data_tables, 0)


class spot_up_shooting(NbaDataProvider):
    def __init__(self, **kwargs):
        url_modifier = 'draftcombinespotshooting'
        NbaDataProvider.__init__(self, url_modifier=url_modifier, default_params=p_draft, **kwargs)

    def data(self):
        return self._get_table_from_data(self._data_tables, 0)


class combine(NbaDataProvider):
    def __init__(self, **kwargs):
        url_modifier = 'draftcombinestats'
        NbaDataProvider.__init__(self, url_modifier=url_modifier, default_params=p_draft, **kwargs)

    def data(self):
        return self._get_table_from_data(self._data_tables, 0)


__all__ = ['anthro', 'agility', 'non_stationary_shooting', 'spot_up_shooting',
           'combine']