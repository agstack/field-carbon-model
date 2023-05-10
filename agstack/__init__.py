class Namespace(object):
    '''
    Dummy class for holding attributes.
    '''
    def __init__(self):
        pass

    def add(self, label, value):
        '''
        Adds a new attribute to the Namespace instance.

        Parameters
        ----------
        label : str
            The name of the attribute
        value : None
            Any kind of value to be stored
        '''
        setattr(self, label, value)
