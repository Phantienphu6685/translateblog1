
# Xây dựng kỹ năng học máy của bạn từ con số 0

> **📖 Bài viết gốc**: [Link](https://aws.amazon.com/vi/blogs/training-and-certification/building-ml-skills-from-zero/)  
> **👤 Tác giả**: Jenny Dassas  
> **📅 Ngày xuất bản**: 01/02/2024  
> **🌐 Nguồn**: AWS Training and Certification Blog  
> **👨‍💻 Người dịch**: Phan Tiến Phú – FCJ Intern  
> **📅 Ngày dịch**: 08/07/2025  
> **⏱️ Thời gian đọc**: 20 phút

---

## 📋 Tóm tắt

Bài viết chia sẻ cách bạn có thể bắt đầu hành trình học máy học (machine learning – ML) từ con số 0, kể cả khi chưa có nền tảng lập trình hoặc khoa học dữ liệu. AWS Training and Certification giới thiệu ba lộ trình học tập được thiết kế phù hợp với từng cấp độ người học – từ người chưa có kinh nghiệm, nhà phát triển, đến chuyên gia dữ liệu. Mỗi lộ trình kết hợp giữa lý thuyết, thực hành, và tài nguyên miễn phí, đồng thời cung cấp các khóa học, tài liệu tương tác và chứng chỉ để giúp bạn phát triển kỹ năng ML theo hướng bền vững.
Ngoài ra, bài viết còn nhấn mạnh vào tầm quan trọng của việc luyện tập với các dự án thực tế và tham gia cộng đồng để học hỏi lẫn nhau. Dù bạn là sinh viên, người chuyển ngành, hay nhà phát triển muốn mở rộng kiến thức, đây là hướng dẫn hữu ích để xây dựng kỹ năng ML một cách có định hướng và hiệu quả.


**🎯 Đối tượng đọc**: Người mới bắt đầu học máy học, sinh viên, nhà phát triển chuyển ngành
  
**📊 Độ khó**: Beginner 
**🏷️ Tags**: machine learning, học máy, kỹ năng AI, AWS Training, entry-level ML

---

## 📚 Mục lục

- [Phần 1: Giới thiệu](#phần-1-giới-thiệu)
- [Phần 2: Hành trình học tập và xây dựng kỹ năng ML](#phần-2-kiến-trúc-hệ-thống)
- [Phần 3: Ứng dụng thực tế và mở rộng với Generative AI](#phần-3-implementation)
- [Kết luận](#kết-luận)
- [Glossary - Thuật ngữ](#glossary---thuật-ngữ)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---
**[AWS Training and Certification Blog](https://aws.amazon.com/blogs/training-and-certification/)**

**Xây dựng kỹ năng học máy của bạn từ con số 0** 

bởi Jenny Dassas | ngày 01 tháng 2 năm 2024 | trong [Alexa](https://aws.amazon.com/blogs/training-and-certification/category/alexa/) , [Amazon Comprehend](https://aws.amazon.com/blogs/training-and-certification/category/artificial-intelligence/amazon-comprehend/) , [Amazon Rekognition](https://aws.amazon.com/blogs/training-and-certification/category/artificial-intelligence/amazon-rekognition/) , [Amazon SageMaker](https://aws.amazon.com/blogs/training-and-certification/category/artificial-intelligence/sagemaker/) , [Amazon Transcribe](https://aws.amazon.com/blogs/training-and-certification/category/artificial-intelligence/amazon-transcribe/) , [Artificial Intelligence](https://aws.amazon.com/blogs/training-and-certification/category/artificial-intelligence/) , [AWS Training and Certification](https://aws.amazon.com/blogs/training-and-certification/category/aws-training-and-certification/), [Best Practices](https://aws.amazon.com/blogs/training-and-certification/category/post-types/best-practices/), [Customer Solutions](https://aws.amazon.com/blogs/training-and-certification/category/post-types/customer-solutions/), [Generative AI](https://aws.amazon.com/blogs/training-and-certification/category/artificial-intelligence/generative-ai/), [Technical How-to](https://aws.amazon.com/blogs/training-and-certification/category/post-types/technical-how-to/) | [Permalink](https://aws.amazon.com/blogs/training-and-certification/building-ml-skills-from-zero/) |  [Chia sẻ](https://aws.amazon.com/vi/blogs/training-and-certification/building-ml-skills-from-zero/#)

Đối với tôi, việc điều hướng bối cảnh công nghệ mà không có nền tảng khoa học máy tính truyền thống là một hành trình đầy thử thách nhưng cũng rất bổ ích. Khoảng hai năm trước, tôi đã chuẩn bị để tham gia [kỳ thi AWS Certified Cloud Practitioner](https://aws.amazon.com/vi/certification/certified-cloud-practitioner/) và nếu bạn hỏi tôi rằng liệu tôi có bao giờ tưởng tượng được việc làm việc với các nhà công nghệ lỗi lạc và một số công ty doanh nghiệp lớn nhất không, tôi sẽ cười vào mặt bạn.

Trong blog này, tôi sẽ chia sẻ hành trình không theo truyền thống của mình đến với sự nghiệp trong lĩnh vực công nghệ và con đường này đã dẫn tôi đến khám phá thế giới phức tạp của học máy (ML). Tôi cũng sẽ chia sẻ lời khuyên và tài nguyên cho những người học muốn thực hiện bước nhảy vọt.

**Về tôi**


![Jenny Dassas](/images/2.prerequisite/jenny-dasass-300x300.jpg) 
*Jenny Dassas*   

Tôi là một cựu sinh viên kỹ thuật hóa học, thất bại, chuyển sang MBA, người đã dành phần lớn sự nghiệp của mình làm việc trong khu vực công. Khi tôi biết về [AWS re/Start](https://aws.amazon.com/vi/training/restart/) , một chương trình chuyên sâu kéo dài 12 tuần dạy các kỹ năng điện toán đám mây, tôi nghĩ mình sẽ thử. Mặc dù không phải là hóa học, nhưng điện toán đám mây nghe có vẻ là một bước ngoặt thú vị vì tôi luôn bị hấp dẫn bởi các lĩnh vực công nghệ khác nhau. Thông qua chương trình, tôi đã có được kinh nghiệm thực tế với các dịch vụ AWS Cloud và phát triển các kỹ năng về Linux, Python, thiết kế cơ sở dữ liệu, v.v.

Nhờ sự chăm chỉ của mình trong chương trình, tôi đã giành được một suất trong [AWS Tech U](https://www.amazon.jobs/content/en/teams/amazon-web-services/tech-u) , một chương trình phát triển lực lượng lao động tăng tốc kéo dài 48 tuần bao gồm đào tạo tại chỗ và học tập dựa trên dự án. Dự án capstone của tôi đã cho phép tôi có được trải nghiệm đầu tiên về ML bằng cách tận dụng Amazon Transcribe, Amazon Comprehend và Amazon Rekognition để phân tích giọng nói và biểu cảm khuôn mặt. Công cụ nội bộ này đã cung cấp phản hồi mang tính xây dựng cho những người nói chuyện được Chứng nhận AWS. Mặc dù tại thời điểm đó, tôi chỉ có thể tham gia vào các dịch vụ ML ở lớp ứng dụng, nhưng trải nghiệm này đã khơi dậy sự quan tâm của tôi trong việc phát triển thêm các kỹ năng của mình. Sự tự tin của tôi trong việc nắm bắt các khái niệm khoa học dữ liệu nâng cao còn khiêm tốn nhưng thông qua sự quyết tâm và học tập liên tục, tôi biết mình có thể mở rộng khả năng của mình.
 
Với vai trò là Quản lý giải pháp khách hàng tại AWS, tôi chịu trách nhiệm giúp khách hàng áp dụng các công cụ và công nghệ của chúng tôi. Đối với việc học theo dự án, tôi được giao làm việc với một công ty Lập kế hoạch nguồn lực doanh nghiệp trong dự án Alexa Smart Properties. Trong khi theo dõi cuộc gọi của khách hàng, Giám đốc quản lý sản phẩm cấp cao đã thảo luận về các chủ đề nâng cao bao gồm các điểm khó khăn với Amazon Lex, các chiến lược tối ưu hóa chi phí cho Amazon Kendra và cơ sở dữ liệu vector. Tôi hiểu biết rất ít về các công nghệ phức tạp này. Sự xuất hiện của trí tuệ nhân tạo (AI) tạo ra càng làm tăng thêm sự bất an của tôi. Tôi quyết tâm nâng cao vai trò của mình và góp phần giải quyết các vấn đề phức tạp thông qua các giải pháp dựa trên dữ liệu, giúp tôi củng cố các kỹ năng ML của mình thông qua việc học tập tại nơi làm việc.

**Hành trình rèn luyện kỹ năng ML của tôi**
Để bắt đầu, tôi đã áp dụng phương pháp học tự định hướng, sử dụng các khóa học trực tuyến, hướng dẫn và các dự án thực hành để thu hẹp khoảng cách kiến ​​thức của mình. [AWS Skill Builder](https://skillbuilder.aws/) có hơn 600 khóa học kỹ thuật số miễn phí trên nhiều lĩnh vực dịch vụ AWS, dành cho người mới bắt đầu đến người học nâng cao, cũng như các tài nguyên học tập thực hành chỉ dành cho đăng ký như AWS Cloud Quest hoặc AWS Jam.

Bạn không cần có nền tảng kỹ thuật hoặc đã trải qua các chương trình chuyên sâu để bắt đầu tìm hiểu về máy học. Đây là một lĩnh vực có lợi cho mọi người ở nhiều vai trò để nâng cao kỹ năng. Tôi bắt đầu với một vài khóa học cơ bản ngắn về ML, mỗi khóa đều thân thiện với người mới bắt đầu.
 - [Giới thiệu về Học máy: Nghệ thuật của khả năng](https://explore.skillbuilder.aws/learn/course/external/view/elearning/11322/introduction-to-machine-learning-art-of-the-possible)
 - [Những điều cần thiết về Machine Learning dành cho những người ra quyết định trong kinh doanh và kỹ thuật](https://explore.skillbuilder.aws/learn/course/external/view/elearning/1404/machine-learning-essentials-for-business-and-technical-decision-makers)
 - [Khám phá Bộ công cụ học máy](https://explore.skillbuilder.aws/learn/course/external/view/elearning/325/exploring-the-machine-learning-toolset)

Sau khi có được hiểu biết cơ bản về học máy, tôi đã sẵn sàng để đào sâu các kỹ năng của mình. Tôi đã đăng ký các khóa đào tạo AWS sau đây, trong đó hướng dẫn các khía cạnh chính của quy trình học máy và các ứng dụng thực tế:

 - [Machine Learning Pipeline trên AWS](https://www.aws.training/SessionSearch?pageNumber=1&courseId=38910&languageId=1) cung cấp trải nghiệm thực tế với quy trình làm việc ML toàn diện trên AWS bao gồm chuẩn bị dữ liệu, đào tạo mô hình, đánh giá và triển khai.
 - [Khoa học dữ liệu thực tế với Amazon SageMaker](https://www.aws.training/SessionSearch?pageNumber=1&courseId=40748&languageId=1) tập trung vào việc xây dựng, đào tạo, điều chỉnh và triển khai các mô hình ML trong SageMaker bằng cách sử dụng các tập dữ liệu thực.
 - [Giải pháp phân tích dữ liệu hàng loạt trên AWS](https://www.aws.training/SessionSearch?pageNumber=1&courseId=91270&languageId=1) đã khám phá các phương pháp hay nhất để thiết kế hệ thống xử lý dữ liệu như quy trình ETL nhằm chuẩn bị dữ liệu cho máy học.

**Học ML bằng cách thực hành**

Càng tìm hiểu về máy học, thông qua các khóa học và làm việc với khách hàng về các trường hợp sử dụng thực tế trong ngành, sự tò mò của tôi càng lớn. Tôi bắt đầu suy nghĩ nghiêm túc về các vấn đề kinh doanh mà ML có thể giúp giải quyết và cách tôi có thể áp dụng các kỹ năng mới của mình để tạo ra tác động. Điều này thúc đẩy tôi bổ sung kiến ​​thức của mình bằng một số dự án nhỏ thực hành sử dụng các tập dữ liệu mẫu, chẳng hạn như xây dựng một mô hình phân loại đơn giản trong SageMaker để dự đoán tình trạng mất khách hàng. Mặc dù cơ bản, những nỗ lực ban đầu này trong việc áp dụng ML đã củng cố sự hiểu biết và sự tự tin của tôi. Học bằng cách làm và bước ra khỏi vùng an toàn đã thúc đẩy các kỹ năng của tôi và cho tôi thấy có bao nhiêu tiềm năng để chuyển đổi các tổ chức bằng ML.

**Chuẩn bị cho AWS Certified Machine Learning- Chuyên ngành**

Kỳ thi [AWS Certified Machine Learning – Specialty](https://aws.amazon.com/certification/certified-machine-learning-specialty/) là một trong những chứng chỉ AWS khó nhất nhưng tôi muốn xác thực các kỹ năng mới tìm thấy của mình bằng cách vượt qua kỳ thi và đạt được chứng chỉ ngành đáng mơ ước này. Bạn có thể nhớ rằng trước đây tôi đã đạt được chứng chỉ cơ bản, AWS Certified Cloud Practitioner. Tôi khuyên mọi người nên bắt đầu với chứng chỉ này và có thể cũng đạt được Chứng chỉ AWS cấp độ Cộng sự trước khi chuyển sang các kỳ thi Chuyên ngành.

AWS không yêu cầu bạn phải chuẩn bị cụ thể trước kỳ thi. Tuy nhiên, tôi khuyên bạn nên làm theo các bước sau để chuẩn bị cho ngày thi:

- Xem lại hướng dẫn thi để hiểu nội dung thi và làm Bộ câu hỏi thực hành chính thức của AWS Certification, có trong [AWS Skill Builder](http://www.skillbuilder.aws/) , để hiểu các câu hỏi theo phong cách thi.
- Tìm hiểu về các chủ đề thi bằng cách đăng ký khóa đào tạo kỹ thuật số trên AWS Skill Builder.
- Chuẩn bị cho kỳ thi bằng cách [đăng ký AWS Skill Builder](https://aws.amazon.com/training/digital/?trk=1a188bd5-9b05-451e-b80b-3e515d98300c&sc_channel=el&trk=d7b1ec2a-31f1-427d-a285-98e21e051bb3&sc_channel=el) để truy cập Khóa học luyện thi tự học (có Tài liệu thực hành). Xem lại các sách trắng và Câu hỏi thường gặp liên quan đến dịch vụ AWS có trên trang thi.
- Xác minh sự sẵn sàng cho kỳ thi của bạn bằng cách làm Bài kiểm tra thực hành chính thức của AWS Certification có sẵn trên AWS Skill Builder khi đăng ký.

**Các khóa học và tài nguyên cụ thể mà tôi đã sử dụng bao gồm:**

- [Chuyên ngành Machine Learning](https://www.coursera.org/specializations/machine-learning-introduction) bao gồm nhiều chủ đề toàn diện, bao gồm hồi quy tuyến tính, thuật toán học máy, mạng nơ-ron, học sâu, mô hình trình tự và các ứng dụng thực tế, cung cấp hiểu biết toàn diện về lĩnh vực này. Tôi đã trải qua điều này hai lần, làm lại tất cả các phòng thí nghiệm thực hành để củng cố kiến ​​thức ứng dụng của mình.
- [Hands-On Machine Learning with Scikit-Learn, Keras và TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) là cuốn sách của Aurélien Géron và nó đã giúp tôi hiểu rõ hơn về một số mô hình đào tạo, bao gồm máy vectơ hỗ trợ, cây quyết định, rừng ngẫu nhiên và phương pháp tổng hợp.
- [Khóa học AWS Certified Machine Learning Specialty](https://www.udemy.com/course/aws-machine-learning/) trang bị cho bạn kiến ​​thức chuyên sâu về ML trên AWS thông qua các bài giảng video, phòng thí nghiệm thực hành và bài kiểm tra thực hành để chuẩn bị đầy đủ cho kỳ thi chuyên ngành và cung cấp cho bạn các kỹ năng ứng dụng để xây dựng, đào tạo và triển khai các mô hình ML.
- [Kế hoạch học máy – AWS Skill Builder](https://explore.skillbuilder.aws/learn/public/learning_plan/view/28/machine-learning-learning-plan?la=cta&cta=topbanner) cung cấp các hướng dẫn tương tác, video và phòng thí nghiệm để mang đến cho bạn trải nghiệm học máy thực tế thông qua việc phát triển mô hình, đào tạo thuật toán và triển khai các dự án trên AWS để giúp bạn thành thạo học máy trên đám mây.
- [Chuỗi video hướng dẫn chuyên sâu về Amazon SageMaker - AWS trên YouTube](https://www.youtube.com/playlist?list=PLhr1KZpdzukcOr_6j_zmSrvYnLUtgqsZz) cung cấp các video hướng dẫn và trình diễn chi tiết từ các chuyên gia về máy học của AWS để cung cấp cái nhìn sâu sắc về các khả năng của SageMaker và hướng dẫn bạn cách xây dựng, đào tạo, điều chỉnh, triển khai và quản lý các mô hình máy học.
- [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/) cung cấp tài liệu toàn diện, mã mẫu và hướng dẫn từng bước để hướng dẫn các nhà phát triển nắm được đầy đủ các chức năng của SageMaker cho từng bước của quy trình học máy, từ chuẩn bị dữ liệu và đào tạo mô hình đến triển khai và giám sát.
- [AWS Machine Learning: Exam Preparation](https://aws.amazon.com/training/learning-paths/machine-learning/exam-preparation/) cung cấp các video hướng dẫn chuyên sâu, bài kiểm tra thực hành và kiểm tra kiến ​​thức để chuẩn bị đầy đủ cho học viên tham gia kỳ thi AWS Certified Machine Learning Specialty bằng cách đề cập đến các khái niệm ML chính và các dịch vụ AWS cần thiết cho kỳ thi theo lộ trình học tập hiệu quả và có cấu trúc.

Vào ngày thi, tôi cảm thấy lo lắng và thậm chí lo rằng mình có thể trượt vì định dạng dài và đầy thử thách. Việc chờ đợi kết quả thật căng thẳng. Nhưng ngày hôm sau, tôi phát hiện ra mình đã đỗ! Sau khi đầu tư rất nhiều thời gian và công sức tự học trong khoảng bảy tháng, việc đỗ kỳ thi đã mang lại cho tôi cảm giác tự hào và thành tựu to lớn. Thật vô cùng bổ ích khi thấy sự tận tâm của mình dẫn đến việc xây dựng thành công khối kiến ​​thức học máy mới này. Tôi có động lực để tiếp tục phát triển nền tảng ML vững chắc này.

**Học tập AI tạo sinh**

Trí tuệ nhân tạo tạo sinh, một lĩnh vực con của Học sâu, có khả năng tự động tạo ra nội dung mới. Không giống như các mô hình truyền thống dựa trên các mẫu và quy tắc được xác định trước, Trí tuệ nhân tạo tạo sinh có thể tự động tạo ra văn bản, hình ảnh và thậm chí là mã giống con người. Khám phá các sắc thái của các mô hình tạo sinh với khóa học kỹ thuật số miễn phí, [Trí tuệ nhân tạo tạo sinh với các mô hình ngôn ngữ lớn](https://www.deeplearning.ai/courses/generative-ai-with-llms/) và [Nền tảng trí tuệ nhân tạo tạo sinh trên AWS](https://www.youtube.com/playlist?list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF) , đã trở thành một phần thú vị trong hành trình học tập của tôi, minh họa cách duy trì sự điều chỉnh theo các công nghệ đang phát triển làm phong phú thêm sự hiểu biết và trình độ của một người.

**Phần kết luận**

Khi ML tiếp tục phát triển bối cảnh CNTT, nhúng vào nhiều ngành và lĩnh vực khác nhau, nó ngày càng trở nên không thể thiếu để thúc đẩy đổi mới, nâng cao hiệu quả và mở ra những khả năng mới cho việc ra quyết định dựa trên dữ liệu. Cam kết học tập liên tục của tôi là không lay chuyển. Tôi háo hức tiếp thu kiến ​​thức và công nghệ mới để trở nên thành thạo trong các lĩnh vực tác động trực tiếp đến vai trò của mình. Kế hoạch trước mắt của tôi là khám phá các khái niệm ML tiên tiến và chuyên về AI có trách nhiệm.

Hành trình của tôi trong ML, bắt đầu từ con số không, là minh chứng cho sức mạnh của việc tự học. Nếu bạn đang bắt đầu một hành trình tương tự, lời khuyên của tôi là hãy bắt đầu với tư duy phát triển, hãy nhớ rằng mọi thử thách đều là bước đệm để thúc đẩy bạn tiến về phía trước. Thế giới ML rất rộng lớn và chào đón những người có xuất thân đa dạng, vì vậy hãy bắt đầu ngay hôm nay. Chúc bạn học tập vui vẻ!

***Blog bổ sung và các nguồn tài nguyên học tập về ML, AI tạo sinh và chuẩn bị chứng chỉ:***
- [Các bước để bắt đầu hành trình lấy chứng chỉ AWS của bạn](https://aws.amazon.com/blogs/training-and-certification/steps-to-start-your-aws-certification-journey/)
- [Hành trình của người học: Từ không có kiến ​​thức về đám mây đến đạt được ba chứng chỉ AWS trong một năm](https://aws.amazon.com/blogs/training-and-certification/from-zero-cloud-knowledge-to-achieving-three-aws-certifications-in-one-year/)
- [Tiêu diệt hội chứng kẻ mạo danh khi chuẩn bị cho kỳ thi Chứng nhận AWS](https://aws.amazon.com/blogs/training-and-certification/slay-imposter-syndrome-while-prepping-for-aws-certification-exams/)
- [Làm thế nào tôi đạt được tất cả sáu chứng chỉ AWS chuyên ngành ngay trong lần thử đầu tiên](https://aws.amazon.com/blogs/training-and-certification/how-i-achieved-all-six-specialty-aws-certifications-on-first-attempt/)
- [Hướng dẫn AWS Ramp-up: Trí tuệ nhân tạo](https://d1.awsstatic.com/training-and-certification/ramp-up_guides/Ramp-Up_Guide_Generative_AI.pdf)
- [Amazon Q – Kế hoạch học tập của trợ lý AI tạo ra](https://explore.skillbuilder.aws/learn/public/learning_plan/view/2207/amazon-q-generative-ai-powered-assistant-learning-plan)
## 📖 Glossary - Thuật ngữ

| English | Tiếng Việt | Định nghĩa |
|---------|------------|------------|
| Machine Learning (ML) | Học máy | Nhánh của trí tuệ nhân tạo cho phép hệ thống học từ dữ liệu và cải thiện theo thời gian mà không cần lập trình rõ ràng. |
| Generative AI | AI sinh nội dung | Loại AI có khả năng tạo ra văn bản, hình ảnh hoặc mã mới bằng cách học từ dữ liệu có sẵn. |
| AWS re/Start | Chương trình AWS re/Start | Khóa đào tạo 12 tuần dành cho người mới bắt đầu học về điện toán đám mây. |
| AWS Tech U | Chương trình AWS Tech U | Chương trình đào tạo kỹ thuật 48 tuần kết hợp học qua dự án và thực hành tại chỗ. |
| Amazon SageMaker | Amazon SageMaker | Dịch vụ giúp xây dựng, huấn luyện và triển khai mô hình học máy trên AWS. |
| Amazon Rekognition | Amazon Rekognition | Dịch vụ phân tích hình ảnh và video bằng học sâu để nhận diện khuôn mặt, vật thể, văn bản,… |
| Amazon Transcribe | Amazon Transcribe | Dịch vụ chuyển giọng nói thành văn bản tự động bằng công nghệ nhận dạng tiếng nói. |
| Amazon Comprehend | Amazon Comprehend | Dịch vụ xử lý ngôn ngữ tự nhiên để trích xuất thông tin từ văn bản như chủ đề, thực thể, cảm xúc,… |
| Amazon Lex | Amazon Lex | Dịch vụ xây chatbot và giao diện hội thoại bằng văn bản và giọng nói. |
| Amazon Kendra | Amazon Kendra | Dịch vụ tìm kiếm thông minh sử dụng học máy để cung cấp kết quả tìm kiếm chính xác và theo ngữ cảnh. |
| Vector Database | Cơ sở dữ liệu vector | Loại cơ sở dữ liệu lưu trữ và tìm kiếm dữ liệu véc-tơ trong các ứng dụng AI và tìm kiếm ngữ nghĩa. |
| ETL Pipeline | Quy trình ETL | Quy trình gồm Trích xuất – Chuyển đổi – Tải dữ liệu, để chuẩn bị dữ liệu cho phân tích hoặc học máy. |
| Customer Churn | Khách hàng rời bỏ | Hiện tượng khách hàng ngừng sử dụng sản phẩm hoặc dịch vụ; có thể dự đoán bằng học máy. |
| Classification Model | Mô hình phân loại | Mô hình học máy dùng để phân chia dữ liệu thành các lớp xác định trước. |
| AWS Skill Builder | AWS Skill Builder | Nền tảng học tập của AWS với các khóa học kỹ thuật số và bài thực hành. |
| AWS Certified Cloud Practitioner | Chứng chỉ AWS Certified Cloud Practitioner | Chứng chỉ nền tảng của AWS dành cho người mới làm quen với đám mây. |
| AWS Certified Machine Learning – Specialty | Chứng chỉ AWS Certified Machine Learning – Specialty | Chứng chỉ chuyên sâu xác nhận kỹ năng xây dựng mô hình học máy trên AWS. |
| Support Vector Machines (SVM) | Máy vector hỗ trợ | Thuật toán học máy có giám sát, dùng để phân loại hoặc hồi quy. |
| Decision Trees | Cây quyết định | Mô hình phân nhánh để đưa ra quyết định dựa trên điều kiện dữ liệu. |
| Random Forests | Rừng ngẫu nhiên | Kỹ thuật học máy sử dụng nhiều cây quyết định để cải thiện độ chính xác. |
| Neural Networks | Mạng nơ-ron | Mô hình học sâu mô phỏng cấu trúc não người, dùng trong nhận diện ảnh, ngôn ngữ,… |
| Sequence Models | Mô hình chuỗi | Mô hình xử lý dữ liệu tuần tự như văn bản, chuỗi thời gian hoặc âm thanh. |
| Responsible AI | AI có trách nhiệm | Cách tiếp cận AI đảm bảo tính minh bạch, công bằng và đạo đức trong thiết kế và triển khai. |


## 🔗 Tài liệu tham khảo

### Tài liệu gốc
- [Original Article](https://aws.amazon.com/vi/blogs/training-and-certification/building-ml-skills-from-zero/): Building your machine learning skills from zero
- [Author's Profile](link): Jenny Dassas – Cựu học viên chương trình AWS re/Start, hiện là Customer Solutions Manager tại AWS
- [Related Articles](https://aws.amazon.com/vi/blogs/training-and-certification/from-zero-cloud-knowledge-to-achieving-three-aws-certifications-in-one-year/): Learner journey: From zero cloud knowledge to achieving three AWS Certifications in one year

### Tài liệu tiếng Việt
- [AWS Documentation VN](https://docs.aws.amazon.com/vi_vn/index.html): Tài liệu AWS tiếng Việt
- [AWS Learning Resources](https://skillbuilder.aws/): Tài nguyên học tập AWS
- [Community Discussions](link): Thảo luận cộng đồng

### Tools và Services
- [Amazon SageMaker](https://aws.amazon.com/vi/sagemaker/): Dịch vụ giúp xây dựng, huấn luyện và triển khai mô hình học máy trên AWS, hỗ trợ học máy toàn trình (end-to-end).
- Amazon Transcribe, Comprehend, Rekognition – Bộ công cụ ML ứng dụng:
    + [Transcribe](https://aws.amazon.com/vi/pm/transcribe/?trk=8c5db4be-d32e-451d-bebe-37d3799d4452&sc_channel=ps&ef_id=CjwKCAjwprjDBhBTEiwA1m1d0oY05NnLoJITV0E0zl8mg9LnpgvfIr1nYSsBh2fj11orHmxFetyAhhoCCGIQAvD_BwE:G:s&s_kwcid=AL!4422!3!652937898467!e!!g!!amazon%20voice%20to%20text%20service!19909696712!151321723407&gad_campaignid=19909696712&gbraid=0AAAAADjHtp9S4EjFhB56lYqtf2wKGe8Y_&gclid=CjwKCAjwprjDBhBTEiwA1m1d0oY05NnLoJITV0E0zl8mg9LnpgvfIr1nYSsBh2fj11orHmxFetyAhhoCCGIQAvD_BwE): chuyển giọng nói thành văn bản
    + [Comprehend](https://aws.amazon.com/vi/comprehend/): xử lý và hiểu ngôn ngữ tự nhiên
    + [Rekognition](https://aws.amazon.com/vi/rekognition/): nhận diện khuôn mặt và phân tích hình ảnh
- [Scikit-learn](https://scikit-learn.org/stable/) – Thư viện học máy mã nguồn mở

---

## 💬 Ghi chú của người dịch

Trong quá trình dịch bài viết này, tôi đặc biệt ấn tượng với hành trình học tập của tác giả từ con số 0 đến khi thi đạt chứng chỉ chuyên sâu AWS Machine Learning.

### Challenges trong quá trình dịch
- **Technical Terms**: Machine Learning Pipeline, Generative AI, Vector Database – là các thuật ngữ cần giải thích kỹ hoặc giữ nguyên. | Giải pháp: tạo bảng thuật ngữ kèm định nghĩa để giữ tính nhất quán.
- **Cultural Context**: Imposter syndrome, non-traditional path, enterprise customers – các khái niệm khá "Mỹ hóa", cần điều chỉnh nhẹ để phù hợp văn hóa Việt.
- **Complex Concepts**: ETL pipelines, model training and tuning, sequence models | Giải pháp: đơn giản hóa ngôn ngữ giải thích mà vẫn giữ đúng ý nghĩa chuyên môn.

### Insights gained
- **Technical Learning**: Hiểu được quy trình học máy từ cơ bản đến nâng cao trong hệ sinh thái AWS. Nắm được các dịch vụ AWS phục vụ ML như SageMaker, Transcribe, Comprehend,…
- **Language Skills**: Luyện dịch thuật ngữ công nghệ và chuyển ngữ văn bản có tính chuyên môn cao.
- **Industry Knowledge**: Thấy rõ lộ trình học và thi chứng chỉ AWS cho ngành ML. Nhận diện được nhu cầu thị trường về kỹ năng học máy và AI hiện đại.

---

## 🤝 Đóng góp và Feedback

Bài dịch này được thực hiện trong khuôn khổ **FCJ Internship Program**. 

**📧 Liên hệ**: phantienphu.it@gmail.com  
**💬 Feedback**: Mọi góp ý để cải thiện chất lượng dịch thuật xin gửi về email trên  
**🔄 Updates**: Bài dịch sẽ được cập nhật dựa trên feedback từ cộng đồng

---

*© 2024 - Bản dịch thuộc về Phan Tiến Phú. Vui lòng credit khi sử dụng.*