# Financial Reporting Summarization

## Vision Statement
Financial professionals are faced with an overwhelming amount of financial literature that needs to be processed and analyzed to arrive at an investment thesis and recommendation, including determining the security to choose, at what price point to buy, the exit strategy, and where to buy in the capital structure. This requires reading through hundreds of pages of text, such as macroeconomic research papers, news, annual and quarterly reports, and earnings transcripts. Once a security is purchased, manual analysis of multi-page performance and risk reports must be conducted. To alleviate this burden, there is a growing interest in applying NLP algorithms to generate concise summaries that focus on specific decision-making information while leaving unnecessary verbiage to the side. This can be achieved by either finding labeled data, generating it through SMEs, or using data extraction methods combined with pre-set templates to generate financial summaries.

Our vision is to create a comprehensive financial reporting solution that caters to the needs of institutional investors, while also making algorithm-generated reports accessible to retail investors. We recognize the vast market potential for such a solution, and our goal is to level the playing field by providing timely and impactful financial information to all. In the US alone, there are an estimated 10,000 institutional investors, and with around 6,000 publicly listed companies, there is a demand for financial statement summaries that runs into the millions per year. With additional reports such as quarterly statements, the potential market size is even larger. Moreover, the retail market is also significant, with over 100 million user names at the six largest brokerages, and assuming an average of 20 companies per portfolio, the demand for financial reports is immense.

## Team Members
This project is done by three people: Dmitry Baron, Haibi Lu, and Viola Pu.

### Dmitry Baron @dmitrybaron
Dmitry Baron is the SME in finance and his responsibilities include:

- Coming up with ideas about supervising outputs and providing guidance about generating better results.
- Generating gold standard samples for model training.
- Working with other SMEs in the industry to evaluate and interpret model results.
- Finding the right keywords for model training.
- Evaluating results and cleaning up final outputs.

## Haibi Lu @hlu17
Haibi Lu's responsibilities include:

- Coming up with modeling approaches.
- Working with customized keyword attention layers.
- Training and fine-tuning BART models.
- Designing a demo website.

## Viola Pu @xuanviola
Viola's responsibilities include:

- Initial data setup and scraping.
- Performing EDA and cleaning up data format.
- Working with HTML format to scrape key information and section segmentation.
- Organizing workflow and generating outputs with the GPT-3.5-turbo model.


