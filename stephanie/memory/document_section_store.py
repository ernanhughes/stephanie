from stephanie.models.document_section import DocumentSectionORM


class DocumentSectionStore:
    def __init__(self, session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "document_section"


    def insert(self, section_dict):
        section = DocumentSectionORM(**section_dict)
        self.session.add(section)
        self.session.commit()
        if self.logger:
            self.logger.log("SectionInserted", section.to_dict())
        return section

    def upsert(self, section_dict):
        """
        Update or insert a document section based on document_id and section_name.
        """
        existing = (
            self.session.query(DocumentSectionORM)
            .filter_by(
                document_id=section_dict["document_id"],
                section_name=section_dict["section_name"],
            )
            .first()
        )

        if existing:
            # Update existing fields
            for key, value in section_dict.items():
                setattr(existing, key, value)
            action = "SectionUpdated"
        else:
            existing = DocumentSectionORM(**section_dict)
            self.session.add(existing)
            action = "SectionInserted"

        self.session.commit()

        if self.logger:
            self.logger.log(action, existing.to_dict())

        return existing


    def get_by_document(self, document_id):
        return (
            self.session.query(DocumentSectionORM)
            .filter_by(document_id=document_id)
            .order_by(DocumentSectionORM.id)
            .all()
        )

    def delete_by_document(self, document_id):
        self.session.query(DocumentSectionORM).filter_by(document_id=document_id).delete()
        self.session.commit()
