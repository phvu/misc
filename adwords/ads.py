from googleads import adwords

adwords_client = adwords.AdWordsClient.LoadFromStorage()

# https://developers.google.com/adwords/api/docs/reference/v201601/ConstantDataService#getagerangecriterion
service = adwords_client.GetService('ConstantDataService', version='v201601')
service.getMobileDeviceCriterion()
