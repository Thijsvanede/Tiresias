import csv
import json

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
