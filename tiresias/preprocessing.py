import csv
import itertools
import json
import numpy as np
import torch
from collections import deque

class Preprocessor():
    """Preprocessor class for loading items"""

    def __init__(self):
        """Initialise preprocessor"""
        # Create loader instance
        self.loader = Loader()

    def load(self, infile, max=float('inf'), min_seq_length=1, decode=False):
        """Load data from given input file

            Parameters
            ----------
            infile : string
                Path to input file from which to load data

            max : float, default='inf'
                Maximum number of events to load from input file

            min_seq_length : int, default=1
                Minimum length of sequences to return

            decode : boolean, default=False
                If True, it decodes data from input file
            """
        # Initialise result
        result = dict()

        # Load items
        for data in self.loader.load(infile, max, decode):
            # Get key
            key = (data.get('source'), data.get('src_ip'))
            # Append data
            result_ = result.get(key, [])
            result_.append(data.get('threat_name'))
            result[key] = result_

        # Return result
        return {k:v for k, v in result.items() if len(v) >= min_seq_length}


class PreprocessLoader(object):

    def __init__(self):
        """Load preprocessed data"""
        # Create loader object
        self.loader = Loader()
        # Create filter for preprocessed data
        self.filter = Filter()

    def load(self, infile, dim_in, dim_out=1, max=float('inf'),
        extract=['threat_name', 'operation', '_id', 'severity', 'confidence'],
        train_ratio=0.5, random=False):
        """Load sequences from input file

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            dim_out : int, default=1
                Dimension of output sequence

            max : float, default=inf
                Maximum number of items to extract

            extract : list
                Fields to extract

            train_ratio : float, default=0.5
                Ratio to train with

            random : boolean, default=False
                Whether to split randomly
            """
        # Load data
        data, encodings = self.load_sequences(infile, dim_in, dim_out, max, extract)
        # Split data
        data = self.split_train_test(data, train_ratio, random)
        # Split data on input and output
        for k, v in data.items():
            for k2, v2 in v.items():
                if k == 'host':
                    data[k][k2] = {'X': v2, 'y': v2}
                else:
                    data[k][k2] = {'X': v2[:, :-dim_out ],
                                   'y': v2[:,  -dim_out:]}

        # Return result
        return data, encodings

    def load_sequences(self, infile, dim_in, dim_out=1, max=float('inf'),
        extract=['threat_name', 'operation', '_id', 'severity', 'confidence']):
        """Load sequences from input file

            Parameters
            ----------
            infile : string
                Path to input file

            dim_in : int
                Dimension of input sequence

            dim_out : int, default=1
                Dimension of output sequence

            max : float, default=inf
                Maximum number of items to extract

            extract : list
                Fields to extract

            Returns
            -------
            data : dict()
                Dictionary of key -> data

            encodings : dict()
                Dictionary of key -> mapping
            """
        # Initialise encodings
        encodings = {k: dict() for k in ['host', 'threat_name']}
        # Initialise output
        result = {k: list() for k in ['host'] + extract}

        # Read data
        data = self.loader.load(infile, max=max, decode=True)

        # Read sequences from data
        for host, datapoint in self.filter.ngrams(data, dim_in+dim_out,
            group=lambda x:      (x.get('source'), x.get('src_ip')),
            key  =lambda x: tuple(x.get(item) for item in extract)):

            # Unpack data
            datapoint = {k: v for k, v in zip(extract, zip(*datapoint))}
            datapoint['host'] = host

            # Store data
            for k, v in datapoint.items():
                # Transform data if necessary
                if k == 'threat_name':
                    for x in v:
                        if x not in encodings[k]: encodings[k][x] = len(encodings[k])
                    v = [encodings[k][x] for x in v]
                if k == 'host':
                    if v not in encodings[k]: encodings[k][v] = len(encodings[k])
                    v = encodings[k][v]
                if k == 'operation':
                    v = [x == 'LOG' for x in v]
                elif k in {'confidence', 'severity'}:
                    v = [int(x) for x in v]

                # Update datapoint
                data_ = result.get(k, list())
                data_.append(v)
                result[k] = data_

        # Get data as tensors
        result = {k: np.asarray(v) if k == '_id' else
                     torch.Tensor(v).to(torch.int64) for k, v in result.items()}

        # Return result
        return result, encodings

    def split_train_test(self, data, train_ratio=0.5, random=False):
        """Split data in train and test sets

            Parameters
            ----------
            data : dict()
                Dictionary of identifier -> array-like of data

            train_ratio : float, default=0.5
                Ratio of training samples

            random : boolean, default=False
                Whether to split randomly
            """
        # Get number of samples
        n_samples = next(iter(data.values())).shape[0]

        # Select training and testing data
        # Initialise training
        i_train = np.zeros(n_samples, dtype=bool)
        if random:
            # Set training data to randomly selected
            i_train[np.random.choice(
                    np.arange(i_train.shape[0]),
                    round(0.5*i_train.shape[0]),
                    replace=False
            )] = True
        else:
            # Set training data to first half
            i_train[:int(n_samples*train_ratio)] = True
        # Testing is everything not in training
        i_test  = ~i_train

        # Split into train and test data
        for k, v in data.items():
            data[k] = {'train': v[i_train], 'test': v[i_test]}

        # Return result
        return data



################################################################################
#                    Object for filtering and grouping data                    #
################################################################################
class Filter(object):
    """Filter object for filtering and grouping json data."""

    def __init__(self):
        """Filter object for filtering and grouping json data."""
        pass

    def groupby(self, data, key):
        """Split data by key

            Parameters
            ----------
            data : iterable
                Iterable to split

            key : func
                Function by which to split data

            Yields
            ------
            key : Object
                Key value of item

            item : Object
                Datapoint of data
            """
        for k, v in itertools.groupby(data, key=key):
            for x in v:
                yield k, x

    def aggregate(self, data, group, key=lambda x: x):
        """Aggregate data by key

            Parameters
            ----------
            data : iterable
                Iterable to aggregate

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            Returns
            -------
            result : dict()
                Dictionary of key -> list of datapoints
            """
        # Initialise result
        result = dict()

        # Loop over datapoints split by key
        for k, v in self.groupby(data, group):
            # Add datapoint
            buffer = result.get(k, [])
            buffer.append(key(v))
            result[k] = buffer

        # Return result
        return result

    def ngrams(self, data, n, group, key=lambda x: x):
        """Aggregate data by key

            Parameters
            ----------
            data : iterable
                Iterable to aggregate

            n : int
                Length of n-gram

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            Returns
            -------
            result : dict()
                Dictionary of key -> list of datapoints
            """
        # Initialise result
        result = dict()

        # Loop over datapoints split by key
        for k, v in self.groupby(data, group):
            # Add datapoint
            buffer = result.get(k, deque())
            buffer.append(key(v))
            # Yield if we find n-gram
            if len(buffer) >= n:
                # Yield buffergroup=lambda x: (x.get('source'), x.get('src_ip')), key=lambda x: x.get('detector_name')
                yield k, tuple(buffer)
                # Remove last item
                buffer.popleft()
            # Store buffer
            result[k] = buffer

    def signatures(self, data):
        """Generate signatures for each host in data

            Parameters
            ----------
            data : dict()
                Dictionary of host -> sequence

            Returns
            -------
            signatures : dict()
                Dictionary of host -> signature
            """
        # Initialise signatures
        signatures = dict()

        # Loop over each host to generate a signature
        for host, sequence in data.items():
            # Generate signature
            signature = "Variable"
            # Set signature
            if len(sequence) == 1:
                signature = "Single           {}".format(sequence[0])
            elif len(set(sequence)) == 1:
                signature = "Single repeating {}".format(sequence[0])

            # Set signature
            signatures[host] = signature

        # Return signatures
        return signatures

    def variable_ngrams(self, data, n, group, key=lambda x: x):
        """Return n-grams in data of length n only for non-trivial hosts

            Parameters
            ----------
            data : data : iterable
                Iterable to aggregate

            n : int
                Length of n-gram

            group : func
                Function by which to split data

            key : func
                Function by which to aggregate data

            Yields
            -------
            host : tuple
                Host identifier

            sequence : list of length n
                List containing datapoints of n-gram
            """
        # Aggregate data
        data = list(data)
        data_ = self.aggregate(data, group=group, key=key)
        # Generate signatures
        signatures = self.signatures(data_)
        # Extract only the hosts for which the signature is variable
        hosts = {k for k, v in signatures.items() if "Variable" in v}

        # Extract n-grams
        for host, sequence in self.ngrams(data, n, group=group, key=key):
            # Check if host in variable hosts
            if host in hosts:
                yield host, sequence



################################################################################
#                       Object for loading ndjson files                        #
################################################################################
class Loader(object):
    """Loader for data from preprocessed files"""

    def load(self, infile, max=float('inf'), decode=False):
        """Load data from given input file

            Parameters
            ----------
            infile : string
                Path to input file from which to load data

            max : float, default='inf'
                Maximum number of events to load from input file

            decode : boolean, default=False
                If True, it decodes data from input file
            """
        # Initialise encoding
        encoding = {}

        # Read encoding file
        with open("{}.encoding.json".format(infile)) as file:
            # Read encoding as json
            encoding = json.load(file)
            # Transform
            for k, v in encoding.items():
                encoding[k] = {str(i): item for i, item in enumerate(v)}

        # Read input file
        with open(infile) as infile:
            # Create csv reader
            reader = csv.DictReader(infile)

            # Read data
            for i, data in enumerate(reader):
                # Break on max
                if i >= max: break

                # Decode data
                if decode:
                    yield {k: encoding.get(k, {}).get(v, v) for k, v in data.items()}
                # Or yield data
                else:
                    # Yield result as ints where possible
                    result = dict()
                    for k, v in data.items():
                        try:
                            result[k] = int(v)
                        except ValueError:
                            result[k] = v
                    yield result



################################################################################
#                       Object for loading ndjson files                        #
################################################################################
class NdJsonLoader(object):
    """Loader for quickly loading data from ndjson files"""

    def __init__(self):
        """Data loader for quickly loading data from ndjson files"""
        pass

    def ndjson(self, file, max=float('inf'), field=None):
        """Load ndjson file

            Parameters
            ----------
            file : string
                File from which to read

            max : float, default=float('inf')
                Maximum number of items to read

            field : string, optional
                If given, only return given field
            """
        with open(file) as file:
            for i, line in enumerate(file):
                if i >= max: break
                data = json.loads(line)
                yield data.get(field, data)

    def ndjson_write(self, file, data):
        """Write ndjson file

            Parameters
            ----------
            file : string
                File to write data to

            data : iterable
                Iterable to write
            """
        with open(file, 'w') as file:
            for d in data:
                d = json.dumps(d)
                file.write(d+'\n')
